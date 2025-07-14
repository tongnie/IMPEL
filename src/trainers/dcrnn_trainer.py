import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from src.utils.logging import get_logger
from src.base.trainer import BaseTrainer
from src.utils import graph_algo


class DCRNN_Trainer(BaseTrainer):
    def __init__(self,
                 unknown_set,
                 known_set,
                 n_m,
                 **args):
        super(DCRNN_Trainer, self).__init__(**args)
        self._optimizer = Adam(self.model.parameters(), self._base_lr, eps=1.0e-3)
        self._lr_scheduler = MultiStepLR(self._optimizer,
                                         self._steps,
                                         gamma=self._lr_decay_ratio)
        self._supports = self._calculate_supports(args['adj_mat'], args['filter_type'])
        self._unknown_set = unknown_set
        self._known_set = known_set
        self._n_m = n_m

    def _calculate_supports(self, adj_mat, filter_type):
        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        supports = []
        if filter_type == "laplacian":
            supports.append(graph_algo.calculate_scaled_laplacian(
                new_adj, lambda_max=None, undirected=True).tocoo())
        elif filter_type == "random_walk":
            supports.append(graph_algo.calculate_random_walk_matrix(new_adj).T)
        elif filter_type == "dual_random_walk":
            supports.append(graph_algo.calculate_random_walk_matrix(new_adj))
            supports.append(graph_algo.calculate_random_walk_matrix(new_adj.T))

        # results = []
        # for support in supports:
        #     # print(type(support))
        #     results.append(self._build_sparse_matrix(
        #         support).cuda())  # to PyTorch sparse tensor
        return supports #results

    def _build_sparse_matrix(self, L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def train_batch(self, X, label, iter):
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports

        self.optimizer.zero_grad()

        ##Unknown sensors for testing##
        X = X[:, :, list(self._known_set), :]  # [B,S,N,C]
        label = label[:, :, list(self._known_set), :]
        # supports = [support[:, list(self._known_set)][list(self._known_set), :] for support in supports]
        supports = [support.tocsr()[list(self._known_set), :][:, list(self._known_set)].tocoo() for support in supports]
        results = []
        for support in supports:
            results.append(self._build_sparse_matrix(
                support).cuda())  # to PyTorch sparse tensor
        supports = results
        ##Masked sensors for inductive training##
        missing_index = np.ones(X.shape)
        for j in range(X.shape[0]):
            missing_mask = np.random.choice(range(0, len(self._known_set)), self._n_m,
                                            replace=False)  # Masked locations
            missing_index[j, :, missing_mask, :] = 0
        missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
        X = X * missing_index
        ###############################
        pred = self.model(X, label, supports, iter)
        pred, label = self._inverse_transform([pred, label])
        loss = self.loss_fn(pred, label, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def evaluate_batch(self, X, label):

        ##Unknown sensors for testing##
        missing_index = np.ones(X.shape)
        missing_index[:, :, list(self._unknown_set), :] = 0
        missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
        X = X * missing_index
        results = []
        for support in self.supports:
            results.append(self._build_sparse_matrix(
                support).cuda())  # to PyTorch sparse tensor
        supports = results
        ###############################

        pred = self.model(X, label, supports)
        pred, label = self._inverse_transform([pred, label])
        return self.loss_fn(pred, label, 0.0).item() * len(label)

    def test_batch(self, X, label):

        ##Unknown sensors for testing##
        missing_index = np.ones(X.shape)
        missing_index[:, :, list(self._unknown_set), :] = 0
        missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
        X = X * missing_index
        results = []
        for support in self.supports:
            results.append(self._build_sparse_matrix(
                support).cuda())  # to PyTorch sparse tensor
        supports = results
        ###############################

        pred = self.model(X, label, supports)
        pred, label = self._inverse_transform([pred, label])
        return pred, label

