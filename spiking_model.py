import torch,time,os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


cfg_cnn = [(2, 128, 1),#AP2
        (128, 128, 1),
        (128, 128, 1),
        (256, 256, 1)
       ]
cfg_fc = [640,64,3]


thresh = 0.4
lens = 0.5
decay = 0.5
num_classes = 10
#batch_size  = 4
num_epochs = 101
learning_rate = 1e-3
#input_dim = 8 * 8 * cfg_cnn[-1][1]
input_dim = 84480
time_window = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()




probs = 0.4
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(SNN_Model, self).__init__()
        in_planes, out_planes, stride = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )

        in_planes, out_planes, stride = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )

        in_planes, out_planes, stride = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )

        in_planes, out_planes, stride = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )


        self.fc1 = nn.Linear(input_dim , cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], )

        # self.fc3.weight.data = self.fc3.weight.data * 0.1

        self.alpha1 = torch.nn.Parameter((1e-3 * torch.ones(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-3 * torch.ones(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((1e-1 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-1 * torch.rand(1, cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((1e-4 * torch.rand(cfg_fc[0], cfg_fc[0])).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((1e-4 * torch.rand(cfg_fc[1], cfg_fc[1])).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.rand(1, input_dim)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)

    def produce_hebb(self):
        hebb1 = torch.zeros(input_dim, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return (hebb1, hebb2)


    def forward(self, input, hebb, win = time_window):
        batch_size = input.shape[0]
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], 180, 240, device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], 90, 120, device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], 45, 60, device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], 45, 60, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        hebb1, hebb2  = hebb

        for step in range(win):
            # k_filter = math.exp(-step / 50)

            x = input[:, :, :, :, step]
            c1_mem, c1_spike = mem_update_conv(self.conv1, F.dropout(x, p=0.25, training=self.training), c1_spike, c1_mem, step)
            x = F.avg_pool2d(c1_spike, 2)
            c2_mem, c2_spike = mem_update_conv(self.conv2, F.dropout(x, p=probs, training=self.training), c2_spike, c2_mem, step)
            x = F.avg_pool2d(c2_spike, 2)
            c3_mem, c3_spike = mem_update_conv(self.conv3, F.dropout(x, p=probs, training=self.training), c3_spike, c3_mem, step)
            x = F.avg_pool2d(c3_spike, 2)
            #c4_mem, c4_spike = mem_update_conv(self.conv4, F.dropout(x, p=probs, training=self.training), c4_spike, c4_mem, step)
            #c5_mem, c5_spike = mem_update_conv(self.conv5, F.dropout(c4_spike, p=probs, training=self.training), c5_spike, c5_mem, step)

            #x = F.avg_pool2d(c3_spike, 2)

            x = x.view(batch_size, -1)

            h1_mem, h1_spike, hebb1 = mem_update(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1, x, h1_spike,
                                                 h1_mem, hebb1)

            h2_mem, h2_spike, hebb2 = mem_update(self.fc2, self.alpha2, self.beta2, self.gamma2, self.eta2, h1_spike,
                                                 h2_spike, h2_mem, hebb2)

            h2_sumspike  = h2_sumspike + h2_spike

        return self.fc3(h2_sumspike/time_window), (hebb1.data, hebb2.data)




def mem_update(fc, alpha, beta, gamma, eta, inputs,  spike, mem,hebb):
    state = fc(inputs)
    mem =  (1 - spike) * mem * decay  + state
    now_spike = act_fun(mem - thresh).float()
    return mem, now_spike, hebb



def mem_update_conv(opts, inputs, spike, mem, t):
    state = opts(inputs)
    mem = (1 - spike) * mem * decay + state
    now_spike = act_fun(mem - thresh).float()
    return mem, now_spike
