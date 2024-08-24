# -*- coding: utf-8 -*-
# @Time    : 2023-05-14
# @Author  : lab
# @desc    :
from torch.autograd import Variable
from thop import profile
from thop import clever_format
from PT_Layer import *
from train_partseg import parse_args


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_points = cfg.num_point
        n_neighbor = cfg.num_neighbor
        # --------------------------------------------------------------------------------------------------------------
        # The first layer of the encoder: Linear + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.encoder_fc1 = nn.Sequential(nn.Linear(cfg.input_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 32))
        self.encoder_transformer1 = TransformerBlock(32, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The second layer of the encoder: Downsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_down2 = TransitionDown(n_points // 4, n_neighbor, [64 // 2 + 3, 64, 64])
        self.encoder_transformer2 = TransformerBlock(64, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The third layer of the encoder: Downsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_down3 = TransitionDown(n_points // 16, n_neighbor, [128 // 2 + 3, 128, 128])
        self.encoder_transformer3 = TransformerBlock(128, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The fourth layer of the encoder: Downsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_down4 = TransitionDown(n_points // 64, n_neighbor, [256 // 2 + 3, 256, 256])
        self.encoder_transformer4 = TransformerBlock(256, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The fifth layer of the encoder: Downsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_down5 = TransitionDown(n_points // 256, n_neighbor, [512 // 2 + 3, 512, 512])
        self.encoder_transformer5 = TransformerBlock(512, n_neighbor)

    def forward(self, x):
        all_point_xyz_feature = []
        # --------------------------------------------------------------------------------------------------------------
        # The first layer of the encoder: Linear + Transformer
        # point_xyz: [B, N, 3]; point_feature: [B, N, 22] -> [B, N, 32]
        # --------------------------------------------------------------------------------------------------------------
        point_xyz = x[..., :3]
        point_feature = x[..., :]
        point_normal = x[..., -3:]
        point_feature = self.encoder_fc1(point_feature)
        point_feature = self.encoder_transformer1(point_xyz, point_feature, point_normal)
        all_point_xyz_feature.append([point_xyz, point_feature, point_normal])
        # --------------------------------------------------------------------------------------------------------------
        # The second layer of the encoder: Downsampling + Transformer
        # point_xyz: [B, N, 3] -> [B, N/4, 3]; point_feature: [B, N, 32] -> [B, N/4, 64]
        # --------------------------------------------------------------------------------------------------------------
        point_xyz, point_feature, point_normal = self.transition_down2(point_xyz, point_feature,
                                                                       point_normal)
        point_feature = self.encoder_transformer2(point_xyz, point_feature, point_normal)
        all_point_xyz_feature.append([point_xyz, point_feature, point_normal])
        # --------------------------------------------------------------------------------------------------------------
        # The third layer of the encoder: Downsampling + Transformer
        # point_xyz: [B, N/4, 3] -> [B, N/16, 3]; point_feature: [N, N/4, 64] -> [B, N/16, 128]
        # --------------------------------------------------------------------------------------------------------------
        point_xyz, point_feature, point_normal = self.transition_down3(point_xyz, point_feature,
                                                                       point_normal)
        point_feature = self.encoder_transformer3(point_xyz, point_feature, point_normal)
        all_point_xyz_feature.append([point_xyz, point_feature, point_normal])
        # --------------------------------------------------------------------------------------------------------------
        # The fourth layer of the encoder: Downsampling + Transformer
        # point_xyz: [B, N/16, 3] -> [B, N/64, 3]; point_feature: [B, N/16, 128] -> [B, N/64, 256]
        # --------------------------------------------------------------------------------------------------------------
        point_xyz, point_feature, point_normal = self.transition_down4(point_xyz, point_feature,
                                                                       point_normal)
        point_feature = self.encoder_transformer4(point_xyz, point_feature, point_normal)
        all_point_xyz_feature.append([point_xyz, point_feature, point_normal])
        # --------------------------------------------------------------------------------------------------------------
        # The fifth layer of the encoder: Downsampling + Transformer
        # point_xyz: [B, N/64, 3] -> [B, N/256, 3]; point_feature: [B, N/64, 256] -> [B, N/256, 512]
        # --------------------------------------------------------------------------------------------------------------
        point_xyz, point_feature, point_normal = self.transition_down5(point_xyz, point_feature,
                                                                       point_normal)
        point_feature = self.encoder_transformer5(point_xyz, point_feature, point_normal)
        all_point_xyz_feature.append([point_xyz, point_feature, point_normal])
        # --------------------------------------------------------------------------------------------------------------
        # Return
        # --------------------------------------------------------------------------------------------------------------
        return all_point_xyz_feature


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_neighbor = cfg.num_neighbor
        # --------------------------------------------------------------------------------------------------------------
        # The first layer of the decoder: Linear + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.decoder_fc1 = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, 512))
        self.decoder_transformer1 = TransformerBlock(512, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The second layer of the decoder: Upsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_up2 = TransitionUp(512, 256, 256)
        self.decoder_transformer2 = TransformerBlock(256, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The third layer of the decoder: Upsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_up3 = TransitionUp(256, 128, 128)
        self.decoder_transformer3 = TransformerBlock(128, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The fourth layer of the decoder: Upsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_up4 = TransitionUp(128, 64, 64)
        self.decoder_transformer4 = TransformerBlock(64, n_neighbor)
        # --------------------------------------------------------------------------------------------------------------
        # The fifth layer of the decoder: Upsampling + Transformer
        # --------------------------------------------------------------------------------------------------------------
        self.transition_up5 = TransitionUp(64, 32, 32)
        self.decoder_transformer5 = TransformerBlock(32, n_neighbor)

    def forward(self, all_point_xyz_feature):
        point_feature = all_point_xyz_feature[-1][1]
        # --------------------------------------------------------------------------------------------------------------
        # The first layer of the decoder: Linear + Transformer
        # point_xyz: [B, N/256, 3]; point_feature: [B, N/256, 512] -> [B, N/256, 512]
        # --------------------------------------------------------------------------------------------------------------
        point_feature = self.decoder_fc1(point_feature)
        point_feature = self.decoder_transformer1(all_point_xyz_feature[-1][0], point_feature,
                                                  all_point_xyz_feature[-1][2])
        # --------------------------------------------------------------------------------------------------------------
        # The second layer of the decoder: Upsampling + Transformer
        # point_feature: [B, N/256, 512] -> [B, N/64, 256]
        # --------------------------------------------------------------------------------------------------------------
        point_feature = self.transition_up2(all_point_xyz_feature[-1][0], point_feature, all_point_xyz_feature[-2][0],
                                            all_point_xyz_feature[-2][1])
        point_feature = self.decoder_transformer2(all_point_xyz_feature[-2][0], point_feature,
                                                  all_point_xyz_feature[-2][2])
        # --------------------------------------------------------------------------------------------------------------
        # The third layer of the decoder: Upsampling + Transformer
        # point_feature: [B, N/64, 256] -> [B, N/16, 128]
        # --------------------------------------------------------------------------------------------------------------
        point_feature = self.transition_up3(all_point_xyz_feature[-2][0], point_feature, all_point_xyz_feature[-3][0],
                                            all_point_xyz_feature[-3][1])
        point_feature = self.decoder_transformer3(all_point_xyz_feature[-3][0], point_feature,
                                                  all_point_xyz_feature[-3][2])
        # --------------------------------------------------------------------------------------------------------------
        # The fourth layer of the decoder: Upsampling + Transformer
        # point_feature: [B, N/16, 128] -> [B, N/4, 64]
        # --------------------------------------------------------------------------------------------------------------
        point_feature = self.transition_up4(all_point_xyz_feature[-3][0], point_feature, all_point_xyz_feature[-4][0],
                                            all_point_xyz_feature[-4][1])
        point_feature = self.decoder_transformer4(all_point_xyz_feature[-4][0], point_feature,
                                                  all_point_xyz_feature[-4][2])
        # --------------------------------------------------------------------------------------------------------------
        # The fifth layer of the decoder: Upsampling + Transformer
        # point_feature: [B, N/4, 64] -> [B, N, 32]
        # --------------------------------------------------------------------------------------------------------------
        point_feature = self.transition_up5(all_point_xyz_feature[-4][0], point_feature, all_point_xyz_feature[-5][0],
                                            all_point_xyz_feature[-5][1])
        point_feature = self.decoder_transformer5(all_point_xyz_feature[-5][0], point_feature,
                                                  all_point_xyz_feature[-5][2])
        # --------------------------------------------------------------------------------------------------------------
        # Return
        # --------------------------------------------------------------------------------------------------------------
        return point_feature


class get_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.fc_pred = nn.Sequential(nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(64, cfg.class_num))

    def forward(self, x):
        all_point_xyz_feature = self.encoder(x)
        point_feature = self.decoder(all_point_xyz_feature)
        result = self.fc_pred(point_feature)
        return result


class get_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, predict, target, weights=None):
        """
        :param predict: (N, C)
        :param target: (N)
        :param weights: (N)
        :param weights:
        :return:
        """
        loss = self.cross_entropy_loss(predict, target)
        if weights is not None:
            loss *= weights
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    args = parse_args()
    input_seg_data = Variable(torch.rand(1, 3000, 22))
    model_seg = get_model(args)

    output_seg = model_seg(input_seg_data)
    print(output_seg.shape)

    # Calculate FLOPs (Floating Point Operations) and parameter count.
    flops, params = profile(model_seg, (input_seg_data,))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"Compute the computational complexity：{flops}, Parameter count：{params}")

    # Calculate the model size
    param_size = sum(p.numel() for p in model_seg.parameters()) * 4
    param_size_mb = param_size / (1024 ** 2)
    print(f"Model size：{param_size_mb:.3f} MB")




