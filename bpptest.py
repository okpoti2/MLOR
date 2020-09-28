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

import argparse
import numpy as np
import torch
from bppgpn import GPN
import sys
from evalSequence import validateSequence

# args
parser = argparse.ArgumentParser(description="3D-BPP GPN test")
parser.add_argument('--size', default=50, help="size of already trained model")
#parser.add_argument('--batch_size', default=1, help='')
parser.add_argument('--random', default=0, help='')
parser.add_argument('--test_size', default=6, help="size of 3D-BPP problem to be solved")
parser.add_argument('--test_steps', default=1, help='')
parser.add_argument('--item_file',type=str,default='data/item-west.txt', help='link to file with item info')
parser.add_argument('--box_file',type=str,default='data/box-west.txt', help='link to file with box/bin info')
args = vars(parser.parse_args())


#B = int(args['batch_size'])
B = 1
size = int(args['size'])
test_size = int(args['test_size'])
item_file = args['item_file']
box_file = args['box_file']
random = int(args['random'])
#n_test = int(args['test_steps'])

load_root ='model/bpp'+str(size)+'.pt'

print('=========================')
print('prepare to test')
print('=========================')
print('Hyperparameters:')
print('model size', size)
print('batch size', B)
print('test size', test_size)
print('load root:', load_root)
print('=========================')
    
# greedy
model = torch.load(load_root).cuda()


sequence_coverage = 0
total_coverage = 0
startval = sys.maxsize
num_features = 3 # length, width and height

sequence_coverage = 0
itemInfo = []
boxInfo = []

def preProcessFile():
    global itemInfo
    global boxInfo
    
    itemInfo = np.loadtxt(item_file,delimiter=',')
    boxInfo = np.loadtxt(box_file,delimiter=',')
    
    outX = np.random.rand(B, len(itemInfo), num_features)
    maxDim, minDim = np.max(itemInfo), np.min(itemInfo)
    denom = maxDim - minDim
    for i in range(len(itemInfo)):
        outX[0,i,:] = [((j-minDim)/denom) for j in itemInfo[i]]
    return outX


if random == 1:
    X = np.random.rand(B, test_size, num_features)
else:
    X = preProcessFile()
X = torch.Tensor(X).cuda()

test_size = X.size(1)
#Start with the item with the max volume as initial item in bin
itemVolumes = [np.prod(X[0,i,:].cpu().data.numpy()) for i in range(len(X[0,:,:]))]
maskedItem = np.argmax(itemVolumes)
mask = torch.zeros(B,test_size).cuda()
mask[[i for i in range(B)],maskedItem] += -np.inf

R = 0
reward = 0
Y = X.view(B,test_size,num_features) # to the same batch size
x = Y[:,0,:]
h = None
c = None
sequence = []
batchedSequence = {b: [maskedItem] for b in range(B)}
sequence.append(maskedItem+1)

for k in range(test_size-1):
    
    output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
    
    idx = torch.argmax(output, dim=1)
    sequence.append(idx.item()+1)
    
    #find items that have been sampled to be added so far
    myIndx = idx.clone().cpu().data.numpy()
    for b in range(len(myIndx)):
         batchedSequence[b].append(myIndx[b])
    
    Y1 = Y[[i for i in range(B)], idx.data].clone()
    if k == 0:
        Y_ini = Y1.clone()
        #mask[[i for i in range(B)],0] +=-np.inf
    if k > 0:
        #reward = torch.norm(Y1-Y0, dim=1)
        tryReward = np.zeros(B)
        for r in range(B):
            tryReward[r] = GPN.coverage(X[r,:,:].cpu().data.numpy(),batchedSequence[r],Y1[r].cpu().data.numpy())
        reward = np.max(tryReward)
        del tryReward
        #print(torch.argmax(reward))

    Y0 = Y1.clone()
    x = Y[[i for i in range(B)], idx.data].clone()
    
    R += reward

    mask[[i for i in range(B)], idx.data] += -np.inf
    
#R += torch.norm(Y1-Y_ini, dim=1)


sequence_coverage += R.mean().item()


total_coverage += sequence_coverage
    

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

    
#sequence.append(0)
print('Add items to bin in the following sequence')
print(sequence, '\n contains duplicates?:',checkIfDuplicates(sequence))
#print(set(sequence))
print('total sequence coverage:', total_coverage)

#Re-arrange item info to matc new sequence
newItemInfo = []
for i in sequence:
    newItemInfo.append(itemInfo[i-1])
newItemInfo = np.array(newItemInfo)


#Evaluate the sequence
vs = validateSequence(newItemInfo,boxInfo)
vs.evaluate(True, sequence)






