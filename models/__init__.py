# -*- coding: utf-8 -*-
# @Time    : 2023-05-14
# @Author  : lab
# @desc    :
# ----------------------------------------------------------------------------------------------------------------------
# When using `importlib.import_module` to call a model, it will be automatically invoked.
# ----------------------------------------------------------------------------------------------------------------------
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'PointTransformer'))
