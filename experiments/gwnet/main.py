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
import torch.nn.functional as F

import torch.nn as nn
import torch
from src.models.gwnet import GWNET
from src.utils.helper import get_dataloader, check_device, get_num_nodes, get_null_value
from src.utils.metrics import masked_mae
from src.trainers.gwnet_trainer import GWNET_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config

def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='gwnet',
                        help='which model to train')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--residual_channels', type=int, default=32)
    parser.add_argument('--dilation_channels', type=int, default=32)
    parser.add_argument('--adpadj', action='store_true', help='whether to use adaptive adj')

    parser.add_argument('--node_dim', type=int, default=32)
    parser.add_argument('--llm_enc_dim', type=int, default=4096)
    parser.add_argument('--num_unknown_nodes', type=int, default=10)
    parser.add_argument('--num_masked_nodes', type=int, default=6)

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.steps = [12000]
    print(args)

    folder_name = '{}-{}-{}-{}-{}'.format(args.n_hidden, args.adpadj, args.aug, args.batch_size, args.base_lr)
    args.log_dir = './logs/{}/{}/{}/'.format(args.dataset,
                                             args.model_name,
                                             folder_name)
    args.num_nodes = get_num_nodes(args.dataset)
    args.null_value = get_null_value(args.dataset)                                            

    
    if args.filter_type in ['scalap', 'identity']:
        args.support_len = 1
    else:
        args.support_len = 2

    args.datapath = os.path.join('./data', args.dataset)
    args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower())
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args


def main():

    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)


    model = GWNET(dropout=args.dropout,
                  supports_len=args.support_len,
                  gcn_bool=True,
                  addaptadj=False,
                  aptinit=None,
                  residual_channels=args.residual_channels,
                  dilation_channels=args.dilation_channels,
                  skip_channels=args.n_hidden * 8,
                  end_channels=args.n_hidden * 16,
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

    result_path = args.result_path + '/' + args.dataset + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #####Masked training####
    n_u = args.num_unknown_nodes
    n_m = args.num_masked_nodes
    rand = np.random.RandomState(42)  # Fixed random output, just an example when seed = 0.
    unknown_set = rand.choice(list(range(0, args.num_nodes)), n_u, replace=False)
    unknown_set = set(unknown_set)
    full_set = set(range(0, args.num_nodes))
    known_set = full_set - unknown_set
    #####Masked training####

    trainer = GWNET_Trainer(model=model,
                            adj_mat=adj_mat,
                            filter_type=args.filter_type,
                            data=data,
                            aug=args.aug,
                            base_lr=args.base_lr,
                            steps=args.steps,
                            lr_decay_ratio=args.lr_decay_ratio,
                            log_dir=args.log_dir,
                            n_exp=args.n_exp,
                            save_iter=args.save_iter,
                            clip_grad_value=args.max_grad_norm,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            device=device,                        
                            result_path=result_path,                     
                            model_name=args.model_name,
                            null_value =args.null_value,
                            unknown_set=unknown_set,
                            known_set=known_set,
                            n_m=n_m,
                            )

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')
    else:
        trainer.test(-1, args.mode)
        if args.save_preds:
            trainer.save_preds(-1)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()