# -*- coding: utf-8 -*-
# @Time    : 2023-05-14
# @Author  : lab
# @desc    :
from einops import rearrange
from PT_Utils import *


class TransitionDown(nn.Module):
    def __init__(self, k, n_neighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0.5, n_neighbor, channels[0], channels[1:], knn=True)

    def forward(self, xyz, points, normals):
        return self.sa(xyz, points, normals)


class TransitionUp(nn.Module):
    def __init__(self, point_few_dim, point_large_dim, dim_out):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(point_few_dim, dim_out),
                                 SwapAxes(),
                                 nn.BatchNorm1d(dim_out),
                                 SwapAxes(),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(point_large_dim, dim_out),
                                 SwapAxes(),
                                 nn.BatchNorm1d(dim_out),
                                 SwapAxes(),
                                 nn.ReLU(inplace=True))
        self.fp = PointNetFeaturePropagation(point_few_dim, [point_few_dim, point_large_dim])

    def forward(self, point_xyz_few, point_feature_few, point_xyz_large, point_feature_large):
        point_feature_few = self.fc1(point_feature_few)
        point_feature_large = self.fc2(point_feature_large)
        new_point_feature_large = self.fp(point_xyz_few, point_feature_few, point_xyz_large,
                                          point_feature_large).transpose(2, 1)
        return new_point_feature_large + point_feature_large


class Local_geometric(nn.Module):
    def __init__(self, transformer_channel, k):
        super().__init__()
        self.transformer_channel = transformer_channel
        self.k = k
        self.fc_position = nn.Sequential(nn.Linear(3, transformer_channel),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(transformer_channel, transformer_channel))
        self.V = nn.Linear(transformer_channel, transformer_channel, bias=False)
        self.fc_attention = nn.Sequential(nn.Linear(transformer_channel, transformer_channel),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(transformer_channel, transformer_channel))
        self.fc_last_local = nn.Sequential(nn.Linear(transformer_channel, transformer_channel),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Linear(transformer_channel, transformer_channel))
        self.bn = nn.Sequential(SwapAxes(),
                                nn.BatchNorm1d(transformer_channel),
                                SwapAxes())
        self.relu = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, feature, xyz, normal):
        # --------------------------------------------------------------------------------------------------------------
        # Calculate neighborhood points
        # --------------------------------------------------------------------------------------------------------------
        dists = Euclidean_Space_distance(xyz, xyz)
        knn_idx = dists.argsort(dim=2)[:, :, :self.k]
        knn_xyz = index_points(xyz, knn_idx)
        knn_dists = torch.gather(dists, 2, knn_idx)[:, :, :, None]
        position_encoding = self.fc_position(knn_xyz - xyz[:, :, None, :])
        # --------------------------------------------------------------------------------------------------------------
        # In Euclidean space, calculate attention based on the distance from points to the tangent plane.
        # --------------------------------------------------------------------------------------------------------------
        q, k, v = xyz, index_points(xyz, knn_idx), index_points(self.V(feature), knn_idx)
        knn_geometric = torch.abs(torch.matmul((q[:, :, None, :] - k), normal[:, :, :, None]))
        attn_geometric = torch.exp(normalize_pytorch_batch(knn_geometric).neg())
        attn_distance = torch.exp(normalize_pytorch_batch(knn_dists).neg())
        attn_multi = normalize_pytorch_batch(attn_geometric * attn_distance)
        attn_multi = attn_multi.expand(-1, -1, -1, self.transformer_channel)
        # --------------------------------------------------------------------------------------------------------------
        # Weighted summation
        # --------------------------------------------------------------------------------------------------------------
        attn = self.fc_attention(attn_multi + position_encoding)
        attn = F.softmax(attn / np.sqrt(attn.size(-1)), dim=-2)
        local_res = torch.einsum('bmnf, bmnf->bmf', attn, v + position_encoding)
        local_res = self.relu(self.bn(self.fc_last_local(local_res))) + feature
        return local_res


class Global_Semantic(nn.Module):
    def __init__(self, channels):
        super(Global_Semantic, self).__init__()
        self.head = 4
        self.q_conv = nn.Conv1d(channels, channels // self.head, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // self.head, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.qh_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = (channels // self.head) ** -0.5

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x, xyz):
        # --------------------------------------------------------------------------------------------------------------
        # Generate the original global attention weights
        # --------------------------------------------------------------------------------------------------------------
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.einsum('bic,bcj->bij', x_q, x_k) * self.scale
        attention = self.softmax(energy)
        # --------------------------------------------------------------------------------------------------------------
        # Use multi-head voting to select key points.
        # --------------------------------------------------------------------------------------------------------------
        x_h = x
        q_h = self.split_heads(x_h, heads=self.head).permute(0, 1, 3, 2)
        k_h = self.split_heads(x_h, heads=self.head)
        energy_h = torch.einsum('bhic,bhcj->bhij', q_h, k_h) * self.scale
        attention_h = self.softmax(energy_h)
        attention_critical = normalize_pytorch_batch_N_N(torch.sum(attention_h, dim=1))
        attention_after = attention * attention_critical
        # --------------------------------------------------------------------------------------------------------------
        # Weighted summation
        # --------------------------------------------------------------------------------------------------------------
        x_s = torch.einsum('bci,bij->bcj', x_v, attention_after)
        x_s = self.act(self.bn(self.trans_conv(x_s)))
        # --------------------------------------------------------------------------------------------------------------
        # Residual
        # --------------------------------------------------------------------------------------------------------------
        x = x + x_s
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def split_heads(x, heads):
        # --------------------------------------------------------------------------------------------------------------
        # (B, H*C, N) -> (B, H, C, N)
        # --------------------------------------------------------------------------------------------------------------
        x = rearrange(x, 'B (H D) N -> B H D N', H=heads).contiguous()
        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_channel, k):
        super().__init__()
        self.local_geometric = Local_geometric(in_channel, k)
        self.global_semantic = Global_Semantic(in_channel)

    def forward(self, xyz, features, normal):
        # --------------------------------------------------------------------------------------------------------------
        # Local transformer
        # --------------------------------------------------------------------------------------------------------------
        local_res = self.local_geometric(features, xyz, normal)
        # --------------------------------------------------------------------------------------------------------------
        # Global transformer
        # --------------------------------------------------------------------------------------------------------------
        global_res = self.global_semantic(local_res, xyz)
        return global_res
