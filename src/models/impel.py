import torch
from torch import nn
from src.base.model import BaseModel
from .mlp import MultiLayerPerceptron
from src.layers.gcn import GCN
import torch.nn.functional as F


class IMPEL(BaseModel):
    def __init__(self,
                 node_dim,
                 input_len,
                 in_dim,
                 embed_dim,
                 output_len,
                 num_layer,
                 llm_enc_dim,
                 supports_len,
                 mp_layers,
                 **args):
        super(IMPEL, self).__init__(**args)
        # attributes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = in_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.num_layer = num_layer
        self.llm_enc_dim = llm_enc_dim
        self.mp_layers = mp_layers
        self.supports_len = supports_len

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        # LLM encoding projection
        self.llm_adapter = nn.Linear(self.llm_enc_dim, self.node_dim)

        # calculate the current adaptive adj matrix once per iteration
        self.supports_len = 1
        # GCN for message passing
        self.gconv = nn.ModuleList()
        for i in range(mp_layers):
            self.gconv.append(
                GCN(self.hidden_dim, self.hidden_dim, dropout=0., support_len=self.supports_len))


    def forward(self, history_data, supports, llm_encoding) -> torch.Tensor:
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)  # [b, c, n, 1]
        time_series_emb = self.time_series_emb_layer(input_data)

        # #####LLM encoding######
        node_emb = []
        # expand node embeddings
        llm_enc = self.llm_adapter(llm_encoding)
        node_emb.append(llm_enc.unsqueeze(0).expand(
            batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb, dim=1)

        # GCN layer
        adp = F.softmax(
            F.gelu(torch.mm(llm_enc, llm_enc.T)), dim=1)
        # new_supports = supports + [adp]  # if in this case, self.supports_len = 3
        new_supports = [adp]

        for i in range(self.mp_layers):
            hidden = self.gconv[i](hidden, new_supports) + hidden

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction
