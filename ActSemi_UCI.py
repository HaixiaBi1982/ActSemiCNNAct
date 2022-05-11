#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:22:51 2020

@author: Haixia Bi
"""
#!coding:utf-8
import os
import torch
from torch import optim
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils.datasets import encode_label,decode_label

from utils import datasets
from utils.ramps import exp_warmup
from utils.datasets import encode_label
from utils.data_utils import NO_LABEL
from utils.config import parse_commandline_args
from utils.data_utils import DataSetWarpper
from utils.data_utils import TwoStreamBatchSampler
from utils.data_utils import TransformTwice as twice
from architectures.arch import arch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import time

from trainer import *
build_model = {    
    'etempensv2_1D': eTempensv2_1D.Trainer,    
    'etempensv2_OnlySup': eTempensv2_OnlySup.Trainer,    
}


def loadData(NumorRatio, ratio, iniLblNum):
    TrPath = 'data-local/UCIHAR/train/Inertial Signals/'    
    Ax = np.loadtxt(os.path.join(TrPath,'body_acc_x_train.txt'))
    Ay = np.loadtxt(os.path.join(TrPath,'body_acc_y_train.txt'))
    Az = np.loadtxt(os.path.join(TrPath,'body_acc_z_train.txt'))
    Gx = np.loadtxt(os.path.join(TrPath,'body_gyro_x_train.txt'))
    Gy = np.loadtxt(os.path.join(TrPath,'body_gyro_y_train.txt'))
    Gz = np.loadtxt(os.path.join(TrPath,'body_gyro_z_train.txt'))  
    xTr = np.array([Ax, Ay, Az, Gx, Gy, Gz])  
    xTr = xTr.astype(np.double)
    xTr = torch.from_numpy(xTr)    
    xTr = xTr.permute(1,2,0)
    xTr = xTr.permute(0,2,1)
    yTr  = np.loadtxt('data-local/UCIHAR/train/y_train.txt')
    yTr = yTr - 1
    yTr = torch.from_numpy(yTr)    
    yTr = yTr.long()

    TsPath = 'data-local/UCIHAR/test/Inertial Signals/'    
    Ax = np.loadtxt(os.path.join(TsPath,'body_acc_x_test.txt'))
    Ay = np.loadtxt(os.path.join(TsPath,'body_acc_y_test.txt'))
    Az = np.loadtxt(os.path.join(TsPath,'body_acc_z_test.txt'))
    Gx = np.loadtxt(os.path.join(TsPath,'body_gyro_x_test.txt'))
    Gy = np.loadtxt(os.path.join(TsPath,'body_gyro_y_test.txt'))
    Gz = np.loadtxt(os.path.join(TsPath,'body_gyro_z_test.txt'))  
    xTs = np.array([Ax, Ay, Az, Gx, Gy, Gz])    
    xTs = xTs.astype(np.double)
    xTs = torch.from_numpy(xTs)    
    xTs = xTs.permute(1,2,0)
    xTs = xTs.permute(0,2,1)
    yTs  = np.loadtxt('data-local/UCIHAR/test/y_test.txt')    
    yTs = yTs - 1
    yTs = torch.from_numpy(yTs)     
    evalset = TensorDataset(xTs, yTs.long())         
    
    # Get the labeled sample indices
    if (NumorRatio == 0):
        lbl_number = iniLblNum
    else:
        lbl_number = ratio*len(yTs)
          
    labeled_idxs = []
    unlabed_idxs = []        
    
    indices = torch.from_numpy(np.array([i for i in range(len(yTr))]))
    indices = indices.cpu().numpy()
    np.random.seed(0)
    np.random.shuffle(indices)
    labeled_idxs.extend(indices[:lbl_number])
    unlabed_idxs.extend(indices[lbl_number:])  
    
    Tridxs = []
    Tridxs += list(labeled_idxs)
    Tridxs += list(unlabed_idxs) 
    
    return xTr, yTr, labeled_idxs, unlabed_idxs, Tridxs, evalset


def create_loaders_Sup(XTr, yTr, labeled_idxs, unlabed_idxs, evalset): 
        
    trainset = TensorDataset(XTr, yTr) 
    trainset_ubl = TensorDataset(XTr[unlabed_idxs], yTr[unlabed_idxs].long()) 

    ## supervised batch loader
    label_sampler = SubsetRandomSampler(labeled_idxs)
    label_batch_sampler = BatchSampler(label_sampler, config.sup_batch_size,
                                       drop_last=True)
    torch.set_default_tensor_type(torch.DoubleTensor)   
    label_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=label_batch_sampler,
                                          num_workers=config.workers,
                                          pin_memory=True)      
    
    ## unsupervised batch loader
    train_loader = label_loader    

    ## unlab_loader_no_shuffle    
    unlab_loader = torch.utils.data.DataLoader(trainset_ubl,
                                           batch_size=len(trainset_ubl), 
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=2*config.workers,
                                           pin_memory=True)   

    
    ## test batch loader   
    eval_loader = torch.utils.data.DataLoader(evalset,  
                                           batch_size=len(evalset),   #batch_size=config.sup_batch_size,
                                           shuffle=False,
                                           num_workers=2*config.workers,
                                           pin_memory=True,
                                           drop_last=False)
    
    loaders = label_loader, train_loader, unlab_loader, eval_loader
    return loaders


def create_loaders_Semi(XTr, yTr, labeled_idxs, unlabed_idxs, Tridxs, evalset):  

    idxTr = torch.from_numpy(np.array([i for i in range(len(yTr))]))
    trainset = TensorDataset(XTr, yTr, idxTr) 
    trainset_ubl = TensorDataset(XTr[unlabed_idxs], yTr[unlabed_idxs].long()) 
    
    # supervised batch loader
    label_sampler = SubsetRandomSampler(labeled_idxs)
    label_batch_sampler = BatchSampler(label_sampler, config.sup_batch_size,
                                       drop_last=True)
    torch.set_default_tensor_type(torch.DoubleTensor)   
    label_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=label_batch_sampler,
                                          num_workers=config.workers,
                                          pin_memory=True)      
    
    # unsupervised batch loader
    unlab_sampler = SubsetRandomSampler(Tridxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, config.usp_batch_size,
                                       drop_last=True)
    train_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=unlab_batch_sampler,
                                          num_workers=config.workers,
                                          pin_memory=True)      
  
    ## unlab_loader_no_shuffle    
    unlab_loader = torch.utils.data.DataLoader(trainset_ubl,
                                           batch_size=len(trainset_ubl), 
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=2*config.workers,
                                           pin_memory=True)          
    
    ## test batch loader 
    eval_loader = torch.utils.data.DataLoader(evalset,
                                           batch_size=len(evalset), 
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=2*config.workers,
                                           pin_memory=True)        
                                   
    loaders=label_loader, train_loader, unlab_loader, eval_loader
    return loaders

def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler


def entropy(probs):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(probs)

    if n_labels <= 1:
        return 0

    n_classes = len(probs)
    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)


def run(config):        
    start = time.time()
    num_classes = 6
    NumorRatio = 0  # 0 stand for number; 1 stand for ratio
    iniLblNum = 100
    ratio = 0.01
    drop_ratio = 0.2
    IncrNum = 100
    Iteration = 3
    feanum = 6
    Criteria = 'BvSB'
    
    for kk in range(20,21):
        output_path = './results/UCI/drop_ratio'+str(drop_ratio)+'/'+Criteria+'/Ite'+str(kk)
        if not os.path.exists(output_path):
            os.makedirs(output_path) 
        print(config)
        print("pytorch version : {}".format(torch.__version__))
        ## create save directory
        if config.save_freq!=0 and not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        
        device = torch.device('cuda:0')
        #device = 'cpu'
        config.drop_ratio=drop_ratio
        ## prepare data
        xTr, yTr, labeled_idxs, unlabed_idxs, Tridxs, evalset = loadData(NumorRatio, ratio, iniLblNum)
        for ite in range(Iteration):    
            if len(labeled_idxs)<100:
                config.sup_batch_size = 10
            elif len(labeled_idxs)<200:
                config.sup_batch_size = 20
            elif len(labeled_idxs)<500:
                config.sup_batch_size = 50
            else:
                config.sup_batch_size = 100              
                
            print(ite)
            print(len(labeled_idxs))
            
            ## prepare architecture
            '''cnn13'''
            net = arch[config.arch](num_classes, config.drop_ratio, feanum)     
            net = net.to(device)
            net = net.double()
            optimizer = create_optim(net.parameters(), config)    
            scheduler = create_lr_scheduler(optimizer, config)

            ## semi-supervised
            if (Criteria == 'BvSB') and ((ite==Iteration-1)):
                config.model='etempensv2_1D'                 
                config.epochs=50
                loaders=create_loaders_Semi(xTr, yTr, labeled_idxs, unlabed_idxs, Tridxs, evalset)
                trainer = build_model[config.model](net, optimizer, device, config)
                outputsAll = trainer.loop(config.epochs, output_path, scheduler, num_classes, *loaders)    
                end = time.time()
                fp=open(os.path.join(output_path,'output.txt'),"a+")
                fp.write('Number of labels:'+str(len(labeled_idxs))+'\r')    
                fp.write('Time:'+str(end-start))
                fp.close()  
                print('Time:'+str(end-start))
                                    
            ## supervised for sample selection
            if (ite<Iteration-1):
                config.arch='cnn13_1D'
                config.model='etempensv2_OnlySup'  
                #config.epochs=100
                net = arch[config.arch](num_classes, config.drop_ratio, feanum)     
                net = net.to(device)
                net = net.double()
                optimizer = create_optim(net.parameters(), config)    
                scheduler = create_lr_scheduler(optimizer, config)            
                loaders = create_loaders_Sup(xTr, yTr, labeled_idxs, unlabed_idxs, evalset)                                  
                trainer = build_model[config.model](net, optimizer, device, config)
                outputsAll = trainer.loop(config.epochs, output_path, scheduler, num_classes, *loaders)    
                fp=open(os.path.join(output_path,'output.txt'),"a+")
                fp.write('Number of labels:'+str(len(labeled_idxs))+'\r\n')    
                fp.close()  
                            
                y_prob_Ulb = np.zeros([len(outputsAll), num_classes])
                for i in range(len(outputsAll)):  
                    exps = np.exp(outputsAll[i,:])
                    y_prob_Ulb[i,:] = exps/np.sum(exps)    
                y_pred_Ulb = np.argmax(y_prob_Ulb, axis=1)                 
    
                if Criteria == 'Entropy':          
                    Entropy = [None]*len(y_prob_Ulb)
                    for k in range(len(y_prob_Ulb)):
                        Entropy[k] = entropy(y_prob_Ulb[k]+1e-10)  
                    sortidx = np.argsort(Entropy) #sort entropies in ascending order
                    sortidx = sortidx[::-1]   #descending   
                    newidx = sortidx[:IncrNum]             
                elif Criteria == 'BvSB':                    
                    y_prob_Ulb.sort()
                    diff_BvSB = y_prob_Ulb[:,num_classes-1]-y_prob_Ulb[:,num_classes-2]
                    sortidx = np.argsort(diff_BvSB)
                    newidx = sortidx[:IncrNum]     
                elif Criteria == 'Random':  
                    indexes = [i for i in range(0, len(y_pred_Ulb))]  
                    np.random.shuffle(indexes)
                    newidx = indexes[:IncrNum]   
               
                newGlbidxs = np.array(unlabed_idxs)[newidx].tolist()
                labeled_idxs = np.concatenate((labeled_idxs, newGlbidxs))  
                unlabed_idxs = (np.setdiff1d(np.array(unlabed_idxs),np.array(newGlbidxs))).tolist()
                #unlabed_idxs = np.delete(unlabed_idxs, newGlbidxs, axis=0)   
               
    
if __name__ == '__main__':
    config = parse_commandline_args()  
        
    config.arch='cnn13_1D'
    config.data_idxs=True
    config.data_root='./data-local'
    config.data_twice=False
    config.dataset='UCI'
    config.drop_ratio=0.2
    config.ema_decay=0.6
    config.ent_weight=None
    config.epochs=50
    config.gamma=None
    config.label_exclude=False
    config.lr=0.1
    config.lr_scheduler='cos'
    config.min_lr=0.0001
    config.mixup_alpha=None
    #config.model='etempensv2_1D'
    config.model='etempensv2_OnlySup'    
    config.momentum=0.9
    config.nesterov=True
    config.num_labels=3300
    config.optim='sgd'
    config.print_freq=20
    config.rampdown_length=50
    config.rampup_length=100
    config.save_dir='./checkpoints'
    config.save_freq=100
    config.soft=None
    config.steps=None
    config.sup_batch_size=50
    config.t1=None
    config.t2=None
    config.usp_batch_size=100
    config.usp_weight=30.0
    config.weight_decay=0.0005
    config.weight_rampup=30
    config.workers=0    
    
    run(config)
