#!/usr/bin/python3
# Author: Hanyang MSL Lab, Industrial Engineering
#
# Copyright (c) 2019 Evans Sowah Okpoti
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from bppgpn import GPN



#------------Helper function-----------------------------
def checkIfDuplicates(listOfElems): 
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

    


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description="3D-BPP GPN with RL")
    parser.add_argument('--size', default=50, help="size of BPP")
    parser.add_argument('--epoch', default=8, help="number of epochs")
    parser.add_argument('--batch_size', default=128, help='')
    parser.add_argument('--train_size', default=1000, help='')
    parser.add_argument('--val_size', default=100, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    args = vars(parser.parse_args())

    size = int(args['size'])
    learn_rate = args['lr']    # learning rate
    B = int(args['batch_size'])    # batch_size
    B_val = int(args['val_size'])    # validation size
    steps = int(args['train_size'])    # training steps
    n_epoch = int(args['epoch'])    # epochs
    save_root ='model/bpp'+str(size)+'.pt'
    
    print('=========================')
    print('Training in progress')
    print('=========================')
    print('Hyperparameters:')
    print('size', size)
    print('learning rate', learn_rate)
    print('batch size', B)
    print('validation size', B_val)
    print('steps', steps)
    print('epoch', n_epoch)
    print('save root:', save_root)
    print('=========================')
    

        
    num_features = 3 # length, width and height
    num_hidden_layers = 128
    
    #Create the graph pointer network model
    model = GPN(n_feature=num_features, n_hidden=num_hidden_layers).cuda()
    # load model
    # model = torch.load(save_root).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    lr_decay_step = 500
    lr_decay_rate = 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                         lr_decay_step), gamma=lr_decay_rate)
    
    # validation data
    X_val = np.random.rand(B_val, size, num_features)

    C = 0     # baseline
    R = 0     # reward

    # R_mean = []
    # R_std = []
    for epoch in range(n_epoch):
        for i in tqdm(range(steps)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
        
            X = np.random.rand(B, size, num_features)        
        
            X = torch.Tensor(X).cuda()
            
            #Select a random batch and find the item with the max volume as initial item in bin
            rndBatch = np.random.randint(1,B)
            itemVolumes = [np.prod(X[rndBatch,i,:].cpu().data.numpy()) for i in range(len(X[rndBatch,:,:]))]
            maskedItem = np.argmax(itemVolumes)
            mask = torch.zeros(B,size).cuda()
            mask[[i for i in range(B)],maskedItem] += -np.inf 
        
            R = 0
            logprobs = 0
            reward = 0
            
            Y = X.view(B,size,num_features)
            x = Y[:,maskedItem,:]
            h = None
            c = None
            batchedSequence = {b: [maskedItem] for b in range(B)}
            for k in range(size-1):
                
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
                
                #find items that have been sampled to be added so far
                myIndx = idx.clone().cpu().data.numpy()
                for b in range(len(myIndx)):
                     batchedSequence[b].append(myIndx[b])
                
                Y1 = Y[[i for i in range(B)], idx.data].clone()
                if k == 0:
                    Y_ini = Y1.clone()
                if k > 0:
                    tryReward = np.zeros(B)
                    for r in range(B):
                        tryReward[r] = GPN.coverage(X[r,:,:].cpu().data.numpy(),batchedSequence[r],Y1[r].cpu().data.numpy())
                    reward = np.max(tryReward)
                    del tryReward
                    #print(reward)
                    
                Y0 = Y1.clone()
                x = Y[[i for i in range(B)], idx.data].clone()
                
                R += reward
                    
                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx.data] + TINY) 
                
                mask[[i for i in range(B)], idx.data] += -np.inf 
                
            R += torch.norm(Y1-Y_ini, dim=1)
            
            
            # self-critic base line
            #mask = torch.zeros(B,size).cuda()
            #Select a random batch and find the item with the max volume
            rndBatch = np.random.randint(1,X.size(0))
            itemVolumes = [np.prod(X[rndBatch,i,:].cpu().data.numpy()) for i in range(len(X[rndBatch,:,:]))]
            maskedItem = np.argmax(itemVolumes)
            mask = torch.zeros(X.size(0),size).cuda()
            mask[[i for i in range(X.size(0))],maskedItem] += -np.inf
            
            C = 0
            baseline = 0
            
            Y = X.view(B,size,num_features)
            x = Y[:,maskedItem,:]
            h = None
            c = None
            batchedSequence = {b: [maskedItem] for b in range(B)}
            for k in range(size-1):
            
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
            
            
                idx = torch.argmax(output, dim=1)    # greedy baseline
                
                #find items that have been sampled to be added so far
                myIndx = idx.clone().cpu().data.numpy()
                for b in range(len(myIndx)):
                     batchedSequence[b].append(myIndx[b])
                
                Y1 = Y[[i for i in range(B)], idx.data].clone()
                if k == 0:
                    Y_ini = Y1.clone()
                if k > 0:
                    baseline = torch.norm(Y1-Y0, dim=1)
                    tryReward = np.zeros(B)
                    for r in range(B):
                        tryReward[r] = GPN.coverage(X[r,:,:].cpu().data.numpy(),batchedSequence[r],Y1[r].cpu().data.numpy())
                    baseline = np.max(tryReward)
                    del tryReward
            
                Y0 = Y1.clone()
                x = Y[[i for i in range(B)], idx.data].clone()
            
                C += baseline
                mask[[i for i in range(B)], idx.data] += -np.inf
        
            C += torch.norm(Y1-Y_ini, dim=1)
        
            gap = (R-C).mean()
            loss = ((R-C-gap)*logprobs).mean()
        
            loss.backward()
            
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()

            if i % 50 == 0:
                print("epoch:{}, batch:{}/{}, reward:{}"
                    .format(epoch, i, steps, R.mean().item()))
                # R_mean.append(R.mean().item())
                # R_std.append(R.std().item())
                
                # greedy validation
                
                sequence_coverage = 0

                X = X_val
                X = torch.Tensor(X).cuda()
                
                #Select a random batch and find the item with the max volume
                rndBatch = np.random.randint(1,X.size(0))
                itemVolumes = [np.prod(X[rndBatch,i,:].cpu().data.numpy()) for i in range(len(X[rndBatch,:,:]))]
                mask = torch.zeros(X.size(0),size).cuda()
                mask[[i for i in range(X.size(0))],maskedItem] += -np.inf
                
                R = 0
                logprobs = 0
                Idx = []
                reward = 0
                
                Y = X.view(B_val, size, num_features)    # to the same batch size
                x = Y[:,0,:]
                h = None
                c = None
                batchedSequence = {b: [maskedItem] for b in range(X.size(0))}
                for k in range(size):
                    
                    output, h, c, hidden_u = model(x=x, X_all=X, h=h, c=c, mask=mask)
                    
                    sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)
                    Idx.append(idx.data)
                
                    Y1 = Y[[i for i in range(B_val)], idx.data]
                    
                    if k == 0:
                        Y_ini = Y1.clone()
                    if k > 0:
                        #reward = torch.norm(Y1-Y0, dim=1)
                        tryReward = np.zeros(X.size(0))
                        for r in range(X.size(0)):
                            tryReward[r] = GPN.coverage(X[r,:,:].cpu().data.numpy(),batchedSequence[r],Y1[r].cpu().data.numpy())
                        reward = np.max(tryReward)
                        del tryReward
            
                    Y0 = Y1.clone()
                    x = Y[[i for i in range(B_val)], idx.data]
                    
                    R += reward
                    
                    mask[[i for i in range(B_val)], idx.data] += -np.inf
            
                R += torch.norm(Y1-Y_ini, dim=1)
                sequence_coverage += R.mean().item()
                print('validating packing coverage:', sequence_coverage)

        print('save model to: ', save_root)
        torch.save(model, save_root)
