import matplotlib.pyplot as plt
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic config
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Test', help='model name, options: [Test, DLinear, SCINet, DLSNet, UNet, Resnet]')
parser.add_argument('--data', type=str, required=True, default='ETTm2', help='dataset type')
parser.add_argument('--moving_avg', type=int, default=24, help='Window size of moving average')

# data loader
parser.add_argument('--root_path', type=str, default='./datasets/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, MS, S]')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') 

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate') 
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='3', help='adjust learning rate')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main
all_mse = []
all_mae = []
for ii in range(args.itr):
    setting = '{}_{}'.format(args.model_id, ii)
    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae, mse_last, mae_last = exp.test(setting)
    all_mse.append(mse)
    all_mae.append(mae)
    torch.cuda.empty_cache()
    
print('mean mse:{}, mean mae:{}, mse_last{}, mae_last{}'.format \
      (np.mean(np.array(all_mse)), np.mean(np.array(all_mae)), np.array(mse_last), np.array(mae_last)))