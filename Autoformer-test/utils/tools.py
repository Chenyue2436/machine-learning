import numpy as np
import torch
import torch.nn as nn # cy!
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    
    # cy_addcode 这是为了让它在笔记本上正常画图，源码中没有这部分也能在google云盘上正常运行
    plt.rcParams['pdf.fonttype'] = 42    
    plt.rcParams['font.family'] = 'Calibri'
    
    plt.savefig(name, bbox_inches='tight')
    
# class StandardScaler():
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def transform(self, data):
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean

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
    
# cy_addcode
class Max_Min_Scaler():
    def __init__(self):
        self._min = 0
        self._max = 1
        
    def fit(self, data):
        self._min = data.min(0) #将data按列求最小
        self._max = data.max(0) #将data按列求最大

    def transform(self, data):
        _min = torch.from_numpy(self._min).type_as(data).to(data.device) if torch.is_tensor(data) else self._min
        _max = torch.from_numpy(self._max).type_as(data).to(data.device) if torch.is_tensor(data) else self._max
        return (data - _min) / (_max - _min)

    def inverse_transform(self, data):
        _min = torch.from_numpy(self._min).type_as(data).to(data.device) if torch.is_tensor(data) else self._min
        _max = torch.from_numpy(self._max).type_as(data).to(data.device) if torch.is_tensor(data) else self._max
        if data.shape[-1] != _min.shape[-1]:
            _min = _min[-1:]
            _max = _max[-1:]
        return (data * (_max - _min)) + _min

class My_loss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, true, prev, way = 0): 
        # pred.ndim == 3:
        # pred.shape:[batch_size, pred_len, c_out]  true.shape:[batch_size, pred_len, c_out]  prev.shape:[batch_size, seq_len, c_out]
        # pred.ndim == 2:
        # pred.shape:[batch_size, pred_len]  true.shape:[batch_size, pred_len]  prev.shape:[batch_size, seq_len]
        
        # MAEloss
        if way == 1:
            return torch.mean(torch.abs(pred - true))
        # MAEloss_penalty
        elif way == 2:
            if pred.ndim == 3:
                prev_new = prev[:,-1,:][:,np.newaxis,:]
            else:
                prev_new = prev[:,-1][:,np.newaxis]
            temp = (pred - prev_new) * (true - prev_new)
            loss1 = torch.abs(pred - true)
            loss2 = self.alpha * (pred - prev_new)**2
            return torch.mean(loss1 + loss2 * (1 - torch.sign(temp)) * 0.5)
        # MAEloss_sparse_penalty
        elif way == 3:
            if pred.ndim == 3:
                pred_new = torch.sort(pred, descending=True)[0]#descending为False，升序，为True，降序
                _max = pred_new[:,:2,:]
                _min = pred_new[:,-2:,:]
                _uniform = pred[:,::11,:]
                pred_new = torch.cat((_max,_min,_uniform),1)
                
                true_new = torch.sort(true, descending=True)[0]#descending为False，升序，为True，降序
                _max = true_new[:,:2,:]
                _min = true_new[:,-2:,:]
                _uniform = true[:,::11,:]
                true_new = torch.cat((_max,_min,_uniform),1) 
                
                prev_new = prev[:,-1,:][:,np.newaxis,:]
            else:
                pred_new = torch.sort(pred, descending=True)[0]#descending为False，升序，为True，降序
                _max = pred_new[:,:2]
                _min = pred_new[:,-2:]
                _uniform = pred[:,::11]
                pred_new = torch.cat((_max,_min,_uniform),1)
                
                true_new = torch.sort(true, descending=True)[0]#descending为False，升序，为True，降序
                _max = true_new[:,:2]
                _min = true_new[:,-2:]
                _uniform = true[:,::11]
                true_new = torch.cat((_max,_min,_uniform),1) 
                
                prev_new = prev[:,-1][:,np.newaxis]
            temp = (pred_new - prev_new) * (true_new - prev_new)
            loss1 = torch.abs(pred_new - true_new)
            loss2 = self.alpha * (pred_new - prev_new)**2
            return torch.mean(loss1 + loss2 * (1 - torch.sign(temp)) * 0.5)
        # 只预测第T+1天相比于第T天的差值
        elif way == 4:
            if pred.ndim == 3:
                pred_c = torch.cat((prev[:, -1:, :],pred[:, :-1, :]), 1)
                true_c = torch.cat((prev[:, -1:, :],true[:, :-1, :]), 1)
                loss = torch.mean(torch.pow(((pred-pred_c) - (true-true_c)), 2))
            else:
                pred_c = torch.cat((prev[:, -1:],pred[:, :-1]), 1)
                true_c = torch.cat((prev[:, -1:],true[:, :-1]), 1)
                loss = torch.mean(torch.pow(((pred-pred_c) - (true-true_c)), 2))
            return loss
        # MSEloss
        else:
            return torch.mean(torch.pow((pred - true), 2))
          
        
