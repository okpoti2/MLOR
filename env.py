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

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import tsp

#city = numItems
#coor = itemDim = l x w x h
#road = seq
#point = (l,w,h)

def get_point(batch,city,coor):
    #output:(batch,city,coor),tensor
    return torch.FloatTensor(np.random.normal(size=(batch,city,coor)))

def get_length(point,road):
    '''point:(batch,city,coor),tensor
        road:(batch,city),numpy
        output:(batch,city),tensor'''
    try:
        length=torch.zeros(torch.IntTensor(road).size())
    except TypeError:
        length=torch.zeros(torch.LongTensor(road).size())
    batch=length.size()[0]
    city=length.size()[1]
    for i in range(batch):
        for j in range(city):
            if j!=city-1:
                length[i,j]=float(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,j+1],:],2)))
            else:
                length[i,j]=float(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,0],:],2)))
    return length

def get_length_sum(point,road):
    '''point:(batch,city,coor),tensor
        road:(batch,city),numpy
        output:(batch),tensor'''
    try:
        dim=road.ndim
    except AttributeError:
        road=road.numpy()
        dim=road.ndim
    if dim==1:
        point=torch.FloatTensor(point)
        city=point.size()[0]
        length=0
        for j in range(city):
            if j!=city-1:
                length+=float(torch.sqrt(torch.sum(torch.pow(point[road[j],:]-point[road[j+1],:],2))))
            else:
                length+=float(torch.sqrt(torch.sum(torch.pow(point[road[j],:]-point[road[0],:],2))))
        print(length)
        return length
    try:
        length=torch.zeros(torch.IntTensor(road).size())
    except TypeError:
        length=torch.zeros(torch.LongTensor(road).size())
    batch=length.size()[0]
    city=length.size()[1]
    for i in range(batch):
        for j in range(city):
            if j!=city-1:
                length[i,j]=float(torch.sqrt(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,j+1],:],2))))
            else:
                length[i,j]=float(torch.sqrt(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,0],:],2))))
    return torch.sum(length,dim=1)

# =============================================================================
# def draw(points,roads):
#     '''point:(batch,city,coor)
#        road:(batch,city)'''
#     if roads.ndim==1:
#         city=len(roads)
#         fig=plt.figure()
#         point=points.numpy()
#         ax=plt.subplot(1,1,1)
#         road=roads
#         for i in range(city-1):
#             ax.plot(point[[road[i],road[i+1]],0],point[[road[i],road[i+1]],1],color='b')
#         ax.plot(point[[road[city-1],road[0]],0],point[[road[city-1],road[0]],1],color='b')
#         fig.show()
#         return 'good'
#     batch=min(roads.shape[0],2)
#     #print(batch)
#     city=len(roads[0])
#     fig=plt.figure()
#     for j in range(batch):
#         ax=plt.subplot(1,batch,j+1)
#         point=points[j].numpy()
#         road=roads[j]
#         for i in range(city-1):
#             ax.plot(point[[road[i],road[i+1]],0],point[[road[i],road[i+1]],1],color='b')
#         ax.plot(point[[road[city-1],road[0]],0],point[[road[city-1],road[0]],1],color='b')
#     fig.show()
# =============================================================================
    
def opt_road(points):
    points=points.numpy()
    if points.ndim==2:
        solution=tsp.tsp(points)
        roads=np.array(solution[1])
        return roads
    batch=points.shape[0]
    roads=[]
    for i in range(batch):
        solution=tsp.tsp(points[i])
        roads.append(solution[1])
    return np.array(roads)
    

