import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time
from torch.fft import rfft,rfftfreq
from torch.fft import irfft

plt.switch_backend('agg')

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0) #将data按列求平均
        self.std = data.std(0)  #将data按列求标准差

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == '1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == '2':
        lr_adjust = {
            2: args.learning_rate * (0.5) , 4: args.learning_rate * (0.5 ** 2), 6: args.learning_rate * (0.5 ** 3),
            8: args.learning_rate * (0.5 ** 4), 10: args.learning_rate * (0.5 ** 5), 12: args.learning_rate * (0.5 ** 6), 
            14: args.learning_rate * (0.5 ** 7), 16: args.learning_rate * (0.5 ** 8), 18: args.learning_rate * (0.5 ** 9)
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate * (0.995 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    if true is not None:
        plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')