import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # raise NotImplementedError的使用感觉很类似于C#中虚函数的效果，
        # 它的意思是如果这个方法没有被子类重写，但是调用了，就会报错。
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            #os.environ是一个环境变量的字典
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    