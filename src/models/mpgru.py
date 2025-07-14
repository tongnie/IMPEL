import torch
from einops import rearrange
from torch import nn

from src.layers.spatial_conv import SpatialConvOrderK
from src.layers.mpgru_layers import MPGRUImputer
from src.base.model import BaseModel
import torch.nn.functional as F
from src.utils import graph_algo
import numpy as np

class MPGRUNet(BaseModel):
    def __init__(self,
                 d_in,
                 d_hidden,
                 d_ff=0,
                 d_u=0,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 support_len=2,
                 layer_norm=False,
                 impute_only_holes=True,
                 **args):
        super(MPGRUNet, self).__init__(**args)


        self.gcgru = MPGRUImputer(input_size=d_in,
                                  hidden_size=d_hidden,
                                  ff_size=d_ff,
                                  u_size=d_u,
                                  n_layers=n_layers,
                                  dropout=dropout,
                                  kernel_size=kernel_size,
                                  support_len=support_len,
                                  layer_norm=layer_norm)
        self.impute_only_holes = impute_only_holes


    def forward(self, x, supports, llm_encoding):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        batch_size, _, num_nodes, _ = x.shape

        x = rearrange(x, 'b s n c -> b c n s')

        # adj = SpatialConvOrderK.compute_support(self.adj, x.device)
        imputation, _ = self.gcgru(x, supports)

        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = rearrange(imputation, 'b c n s -> b s n c')

        return imputation



