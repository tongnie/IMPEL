import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader, check_device, get_num_nodes, get_null_value
from src.utils.metrics import masked_mae
from src.models.ignnk import IGNNK
from src.trainers.ignnk_trainer import IGNNK_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config

def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='ignnk',
                        help='which model to train')
    parser.add_argument('--n_filters', type=int, default=0,
                        help='number of hidden units')
    parser.add_argument('--filter_type', type=str, default='doubletransition')

    parser.add_argument('--time_dimension', type=int, default=24)
    parser.add_argument('--hidden_dimnesion', type=int, default=64)
    parser.add_argument('--order', type=int, default=2)

    parser.add_argument('--source_data', type=str, default='Delivery_SH')
    parser.add_argument('--target_data', type=str, default='Delivery_HZ')
    parser.add_argument('--num_unknown_nodes', type=int, default=10)  # 5 for JL and 10 for others

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.steps = [10, 20, 30, 40, 50]
    print(args)

    # first pretraining the model on the source data
    folder_name_source = '{}-{}'.format(args.model_name, args.aug)
    args.log_dir_pretrained = './logs/{}/{}/{}/'.format(args.source_data,
                                                        args.model_name,
                                                        folder_name_source)

    args.num_nodes = get_num_nodes(args.target_data)
    args.null_value = get_null_value(args.target_data)


    if args.filter_type in ['scalap', 'identity']:
        args.support_len = 1
    else:
        args.support_len = 2

    args.datapath = os.path.join('./data', args.target_data)
    args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.target_data.lower())
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args


def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)


    model = IGNNK(
                 time_dimension=args.time_dimension,
                 hidden_dimnesion=args.hidden_dimnesion,
                 order=args.order,
                 name=args.model_name,
                 dataset=args.dataset,
                 device=device,
                 num_nodes=args.num_nodes,
                 seq_len=args.seq_len,
                 horizon=args.horizon,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 )

    data = get_dataloader(args.datapath,
                          args.batch_size,
                          args.input_dim,
                          args.output_dim)

    
    result_path = args.result_path + '/' + args.target_data + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #####Masked training####
    n_u = args.num_unknown_nodes
    rand = np.random.RandomState(42)  # Fixed random output, just an example when seed = 0.
    unknown_set = rand.choice(list(range(0, args.num_nodes)), n_u, replace=False)
    unknown_set = set(unknown_set)
    full_set = set(range(0, args.num_nodes))
    known_set = full_set - unknown_set
    #####Masked training####


    trainer = IGNNK_Trainer(model=model,
                            adj_mat=adj_mat,
                            filter_type=args.filter_type,
                            data=data,
                            aug=args.aug,
                            base_lr=args.base_lr,
                            steps=args.steps,
                            lr_decay_ratio=args.lr_decay_ratio,
                            log_dir=args.log_dir_pretrained,
                            n_exp=args.n_exp,
                            save_iter=args.save_iter,
                            clip_grad_value=args.max_grad_norm,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            device=device,
                            model_name=args.model_name,
                            result_path=result_path,
                            null_value =args.null_value,
                            unknown_set=unknown_set,
                            known_set=known_set,
                            n_m=0,
                            )


    trainer.test(-1, 'test')



if __name__ == "__main__":
    main()