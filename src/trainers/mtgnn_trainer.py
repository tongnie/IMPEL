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
from src.utils import metrics as mc
import pandas as pd
import csv


class MTGNN_Trainer(BaseTrainer):
    def __init__(self,
                 unknown_set,
                 known_set,
                 n_m,
                 **args):
        super(MTGNN_Trainer, self).__init__(**args)
        self._optimizer = Adam(self.model.parameters(), self._base_lr)
        self._supports = self._calculate_supports(args['adj_mat'], args['filter_type'])
        self._unknown_set = unknown_set
        self._known_set = known_set
        self._n_m = n_m
        
    def _calculate_supports(self, adj_mat, filter_type):
        # num_nodes = adj_mat.shape[0]
        # new_adj = adj_mat + np.eye(num_nodes)
        
        # if filter_type == "scalap":
        #     supports =[graph_algo.calculate_scaled_laplacian(new_adj).todense()]
        # elif filter_type == "normlap":
        #     supports =[graph_algo.calculate_normalized_laplacian(
        #         new_adj).astype(np.float32).todense()]
        # elif filter_type == "symnadj":
        #     supports =[graph_algo.sym_adj(new_adj)]
        # elif filter_type == "transition":
        #     supports =[graph_algo.asym_adj(new_adj)]
        # elif filter_type == "doubletransition":
        #     supports =[graph_algo.asym_adj(new_adj),
        #                graph_algo.asym_adj(np.transpose(new_adj))]
        # elif filter_type == "identity":
        #     supports =[np.diag(np.ones(new_adj.shape[0])).astype(np.float32)]
        # else:
        #     error = 0
        #     assert error, "adj type not defined"
        # supports = [torch.tensor(i).cuda() for i in supports]
        return adj_mat.astype(np.float32) #supports


    # Rewrite the training and testing procedure for masked training
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
        supports = supports[:, list(self._known_set)][list(self._known_set), :]
        ##Masked sensors for inductive training##
        missing_index = np.ones(X.shape)
        for j in range(X.shape[0]):
            missing_mask = np.random.choice(range(0, len(self._known_set)), self._n_m, replace=False)  # Masked locations
            missing_index[j, :, missing_mask, :] = 0
        missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
        X = X * missing_index
        ###############################


        pred = self.model(X, supports)
        pred, label = self._inverse_transform([pred, label])

        loss = self.loss_fn(pred, label, self.null_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")
        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, data in enumerate(self.data['train_loader']):
                (X, label) = data
                X, label = self._check_device([X, label])
                train_losses.append(self.train_batch(X, label, iter))
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                                                                                  self._max_epochs,
                                                                                                  iter,
                                                                                                  np.mean(train_losses),
                                                                                                  val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch

            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                      '{:.1f}s'.format(epoch,
                                       self._max_epochs,
                                       iter,
                                       np.mean(train_losses),
                                       val_loss,
                                       new_lr,
                                       (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for data in self.data['val_loader']:
                (X, label) = data
                X, label = self._check_device([X, label])

                ##Unknown sensors for testing##
                missing_index = np.ones(X.shape)
                missing_index[:, :, list(self._unknown_set), :] = 0
                missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
                X = X * missing_index
                ###############################

                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mae = self.loss_fn(preds, labels, self.null_value).item()
        return mae

    def test_batch(self, X, label):
        pred = self.model(X, self.supports)
        pred, label = self._inverse_transform([pred, label])
        return pred, label

    def test(self, epoch, mode='test'):
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []

        start_time = time.time()

        with torch.no_grad():
            self.model.eval()
            for _, data in enumerate(self.data[mode + '_loader']):
                (X, label) = data
                X, label = self._check_device([X, label])

                ##Unknown sensors for testing##
                missing_index = np.ones(X.shape)
                missing_index[:, :, list(self._unknown_set), :] = 0
                missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
                X = X * missing_index
                ###############################

                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        end_time = time.time()

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        amae = []
        armse = []

        for i in range(self.model.horizon):
            pred = preds[:, i]
            real = labels[:, i]
            metrics = mc.compute_all_metrics(pred, real, self.null_value)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, teat_time {:.1f}s'
            print(log.format(i + 1, metrics[0], metrics[1], (end_time - start_time)))
            amae.append(metrics[0])
            armse.append(metrics[1])

        log = 'On average over {} horizons, Average Test MAE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(self.model.horizon, np.mean(amae), np.mean(armse)))

        csv_path = self.result_path + '/{}.csv'.format(self.model_name)
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=['hp', 'end_time', 'time',
                                       'mae', 'rmse'])
            df.to_csv(csv_path, index=False)

        with open(csv_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [self.hp, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), round(end_time - start_time, 2),
                        np.mean(amae), np.mean(armse)]
            csv_write.writerow(data_row)

        return np.mean(amae)
