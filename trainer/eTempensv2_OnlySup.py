# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:15:18 2020

@author: bhxwzq
"""

#!coding:utf-8
import torch
from torch.nn import functional as F
import pandas as pd
import os
import datetime
from pathlib import Path
from itertools import cycle
from collections import defaultdict
import numpy as np
from utils.loss import mse_with_softmax
from utils.ramps import exp_rampup
from utils.datasets import decode_label
from utils.data_utils import NO_LABEL
from sklearn.metrics import f1_score

    
class Trainer:

    def __init__(self, model, optimizer, device, config):
        print('Tempens-v2 1D with epoch pseudo labels')
        self.model     = model
        self.optimizer = optimizer
        self.ce_loss   = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        self.mse_loss  = mse_with_softmax # F.mse_loss 
        self.save_dir  = '{}-{}_{}-{}_{}'.format(config.arch, config.model,
                          config.dataset, config.num_labels,
                          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_dir  = os.path.join(config.save_dir, self.save_dir)
        self.device      = device
        self.usp_weight  = config.usp_weight
        self.ema_decay   = config.ema_decay
        self.rampup      = exp_rampup(config.rampup_length)
        self.save_freq   = config.save_freq
        self.print_freq  = config.print_freq
        self.epoch       = 0
        self.start_epoch = 0
                
    def train_iteration(self, label_loader, unlab_loader, output_path, print_freq):
        loop_info = defaultdict(list)
        batch_idx, label_n = 0, 0
        for idhyx, (label_x, label_y) in enumerate(label_loader):            
            label_x, label_y = label_x.to(self.device), label_y.to(self.device)
            #unlab_x, unlab_y = unlab_x.to(self.device), unlab_y.to(self.device)
            ##=== decode targets of unlabeled data ===
            #self.decode_targets(unlab_y)
            lbs = label_x.size(0)

            ##=== forward ===
            outputs = self.model(label_x)
            outputs = outputs.to(self.device)
            loss = self.ce_loss(outputs, label_y.long()).float()
            loop_info['lSup'].append(loss.item())

#            ##=== Semi-supervised Training Phase ===
#            unlab_outputs    = self.model(unlab_x)
#            iter_unlab_pslab = self.epoch_pslab[udx.long()]
#            uloss  = self.mse_loss(unlab_outputs, iter_unlab_pslab)
#            uloss *= self.rampup(self.epoch)*self.usp_weight
#            loss  += uloss
#            loop_info['uTmp'].append(uloss.item())
#            ## update pseudo labels
#            with torch.no_grad():
#                if self.epoch==0:
#                    self.epoch_pslab[udx.long()] = unlab_outputs.long().clone().detach() ##Mod
#                else:
#                    self.epoch_pslab[udx.long()] = unlab_outputs.double().clone().detach() ##Mod
            ## bachward 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            batch_idx, label_n = batch_idx+1, label_n+lbs
            loop_info['lacc'].append(label_y.eq(outputs.max(1)[1]).double().sum().item()) 
            #loop_info['uacc'].append(unlab_y.eq(unlab_outputs.max(1)[1].int()).double().sum().item())
            loop_info['lF1'].append(f1_score(label_y.cpu(), outputs.max(1)[1].int().cpu(), average='macro'))   
            #loop_info['uF1'].append(f1_score(unlab_y, unlab_outputs.max(1)[1].int(), average='macro'))
            if int(print_freq)>0 and (int(batch_idx)%int(print_freq))==0:
                print(f"[sup train][{batch_idx:<3}]", self.gen_info(loop_info, lbs))
        # temporal ensemble
        #self.update_ema_predictions() # update every epoch
        print(">>>[sup train]", self.gen_info(loop_info, label_n, False))
        #New added print
        #output_path = 'C:/research/gitrepo/semisupervision_attn_cnn/results/USCHAD/'
        fp=open(os.path.join(output_path,'output.txt'),"a+")
        fp.write('sup Train '+str(self.epoch)+': '+self.gen_info(loop_info, label_n, False)+'\n')
        fp.close()
        return loop_info, label_n

    def test_iteration(self, data_loader, output_path, ep, num_classes, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0        
        outputsAll = np.empty((0, num_classes)) 
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            ##=== decode targets ===
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            outputs = self.model(data)
            #outputs = outputs.to(self.device)
            loss = self.ce_loss(outputs, targets)
            loop_info['lloss'].append(loss.item())

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).double().sum().item())
            loop_info['lF1'].append(f1_score(targets.cpu(), outputs.max(1)[1].int().cpu(), average='macro'))
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[sup test][{batch_idx:<3}]", self.gen_info(loop_info, lbs))
            outputsAll = np.concatenate((outputsAll, outputs.cpu()),axis=0)  
            
        outputsAll = pd.DataFrame(outputsAll)
        #outputsAll.to_csv(os.path.join(output_path,'outputs'+str(ep)+'.csv'))
        print(">>>[sup test]", self.gen_info(loop_info, label_n, False))   
        #New added print
        #output_path = 'C:/research/gitrepo/semisupervision_attn_cnn/results/USCHAD/'
        fp=open(os.path.join(output_path,'output.txt'),"a+")
        fp.write('sup Test '+str(self.epoch)+': '+self.gen_info(loop_info, label_n, False)+'\n')
        fp.close()
        return loop_info, label_n, outputsAll   

    def validate_iteration(self, data_loader, output_path, ep, num_classes, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0        
        outputsAll = np.empty((0, num_classes)) 
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            ##=== decode targets ===
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            outputs = self.model(data)
            loss = self.ce_loss(outputs, targets)
            loop_info['lloss'].append(loss.item())

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).double().sum().item())
            loop_info['lF1'].append(f1_score(targets.cpu(), outputs.max(1)[1].int().cpu(), average='macro'))
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[sup val][{batch_idx:<3}]", self.gen_info(loop_info, lbs))
            outputsAll = np.concatenate((outputsAll, outputs.cpu()),axis=0)  
            
        #pdoutputsAll = pd.DataFrame(outputsAll)
        #outputsAll.to_csv(os.path.join(output_path,'outputs'+str(ep)+'.csv'))
        print(">>>[sup val]", self.gen_info(loop_info, label_n, False))   
        #New added print
        #output_path = 'C:/research/gitrepo/semisupervision_attn_cnn/results/USCHAD/'
        fp=open(os.path.join(output_path,'output.txt'),"a+")
        fp.write('sup val '+str(self.epoch)+': '+self.gen_info(loop_info, label_n, False)+'\n')
        fp.close()
        return loop_info, label_n, outputsAll    

    def train(self, label_loader, unlab_loader, output_path, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, unlab_loader, output_path, print_freq)

    def test(self, data_loader, output_path, ep, num_classes, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, output_path, ep, num_classes, print_freq)

    def validate(self, data_loader, output_path, ep, num_classes, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            return self.validate_iteration(data_loader, output_path, ep, num_classes, print_freq)

    def update_ema_predictions(self):
        """update every epoch"""
        self.ema_pslab = (self.ema_decay*self.ema_pslab) + (1.0-self.ema_decay)*self.epoch_pslab.double()
        self.epoch_pslab = self.ema_pslab / (1.0 - self.ema_decay**((self.epoch-self.start_epoch)+1.0))

    def loop(self, epochs, output_path, scheduler, num_classes, label_data, unlab_data, unlab_data_nosh, test_data):
        ## main process
        best_info, best_acc, n = None, 0., 0
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None: scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(label_data, unlab_data, output_path, self.print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            info, n, outputsAll = self.test(test_data, output_path, ep, num_classes, self.print_freq)
            acc     = sum(info['lacc']) / n
            if acc>best_acc: best_info, best_acc = info, acc
            ## save model
            if self.save_freq!=0 and (ep+1)%self.save_freq == 0:
                self.save(ep)
                
        print("------ validation: {} ------".format(ep))
        info, n, outputsAll = self.validate(unlab_data_nosh, output_path, ep, num_classes, self.print_freq)                          
                
        print(f">>>[best]", self.gen_info(best_info, n, False))
        fp=open(os.path.join(output_path,'output.txt'),"a+")
        fp.write('best '+': '+self.gen_info(best_info, n, False))
        fp.close()      
        return outputsAll

    def create_soft_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype=='rand': 
             pslab = torch.randint(0, n_classes, (n_samples,n_classes))
        elif dtype=='zero':
             pslab = torch.zeros(n_samples, n_classes)
        else:
             raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab

    def decode_targets(self, targets):
        label_mask = targets.ge(0)
        unlab_mask = targets.le(NO_LABEL)
        targets[unlab_mask] = decode_label(targets[unlab_mask])
        return label_mask, unlab_mask

    def gen_info(self, info, lbs, iteration=True):
        ret = []
        nums = {'l': lbs, 'a': lbs}
        if info != None:
            for k, val in info.items():
                n = nums[k[0]]
                v = val[-1] if iteration else sum(val)
                if (k[-1]=='c'):
                    s = f'{k}: {v/n:.3%}' 
                elif (k[-1]=='1'):
                    s = f'{k}: {v:.3f}' 
                else:
                    s = f'{k}: {v:.5f}'
                ret.append(s)
        return '\t'.join(ret)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_target = model_out_path / "model_epoch_{}.pth".format(epoch)
            torch.save(state, save_target)
            print('==> save model to {}'.format(save_target))
