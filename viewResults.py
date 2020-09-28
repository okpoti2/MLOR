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

import matplotlib.pyplot as plt
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

class viewResults:
    
    def __init__(self,itemInfo, boxInfo, bin_item_assign):
        self.itemInfo = itemInfo
        self.boxInfo = boxInfo
        self.binItemAssign = bin_item_assign
        self.x, self.y, self.z = [],[],[]
        self.lx, self.ly, self.lz = [],[],[]
        self.wx, self.wy, self.wz = [],[],[]
        self.hx, self.hy, self.hz = [],[],[]
        
    def setSpatialCoordinates(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        
    def setLengthAlign(self,lx,ly,lz):
        self.lx, self.ly, self.lz = lx, ly, lz
        
    def setWidthAlign(self,wx,wy,wz):
        self.wx, self.wy, self.wz = wx, wy, wz
        
    def setHeightAlign(self,hx,hy,hz):
        self.hx, self.hy, self.hz = hx, hy, hz
        
    def plotOneResult(self,binId,itemIdList):
        
         
        base_colors = ['#d3d3d3', '#2f4f4f', '#a52a2a', '#2e8b57', '#808000', '#000080', '#ff0000', '#ff8c00', '#ba55d3', '#00ff7f', '#ff1493']
        
        l = itemIdList[0]

        # prepare box/bin               
        x, y, z = np.indices((int(self.boxInfo[binId][0]), int(self.boxInfo[binId][1]), int(self.boxInfo[binId][2])))
        
        #add first item
        pos_x = (self.lx[l]*self.itemInfo[l][0] + self.wx[l]*self.itemInfo[l][1] + self.hx[l]*self.itemInfo[l][2])
        pos_y = (self.ly[l]*self.itemInfo[l][0] + self.wy[l]*self.itemInfo[l][1] + self.hy[l]*self.itemInfo[l][2])
        pos_z = (self.lz[l]*self.itemInfo[l][0] + self.wz[l]*self.itemInfo[l][1] + self.hz[l]*self.itemInfo[l][2])
        
        
        firstItem = ((int(self.x[l])<=x) & (x<pos_x+int(self.x[l]))) & ((int(self.y[l])<=y) & (y<pos_y+int(self.y[l]))) & ((int(self.z[l])<=z) & (z<pos_z+int(self.z[l])))
        #print(self.x[0],',',self.y[0],',',self.z[0])
        voxels = firstItem
        #print(self.x[l],',',self.y[l],',',self.z[l],'-------',self.itemInfo[l])
        
        items_dict = {}
        for k in range(1,len(itemIdList)):
            i = itemIdList[k]
            pos_x = (self.lx[i]*self.itemInfo[i][0] + self.wx[i]*self.itemInfo[i][1] + self.hx[i]*self.itemInfo[i][2])
            pos_y = (self.ly[i]*self.itemInfo[i][0] + self.wy[i]*self.itemInfo[i][1] + self.hy[i]*self.itemInfo[i][2])
            pos_z = (self.lz[i]*self.itemInfo[i][0] + self.wz[i]*self.itemInfo[i][1] + self.hz[i]*self.itemInfo[i][2])
            nextItem = ((int(self.x[i])<=x) & (x<pos_x+int(self.x[i]))) & ((int(self.y[i])<=y) & (y<pos_y+int(self.y[i]))) & ((int(self.z[i])<=z) & (z<pos_z+int(self.z[i])))
            print(self.x[i],',',self.y[i],',',self.z[i],'-------',self.itemInfo[i])
            #print('---',pos_x,',',pos_y,',',pos_z)
            items_dict[k] = nextItem
            voxels |=nextItem
            #colors[nextItem] = base_colors[i]
        #print(self.x)
        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        colors[firstItem] = base_colors[0]
        j = 1
        for key in items_dict.keys():
            colors[items_dict[key]] = base_colors[j]
            j+=1
        # and plot everything
        c = max(int(self.boxInfo[binId][0]),int(self.boxInfo[binId][1]),int(self.boxInfo[binId][2]))
        fig = plt.figure(figsize=(c,int(c*0.85)))
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        #ax.view_init(0, 90)
        plt.show()

if __name__ == "__main__":
    # prepare some coordinates
    x, y, z = np.indices((35, 20, 10))
    itemInfo = [[25,8,6],[20,10,5],[16,7,3],[15,12,6]]
    i=0
    j=1
    k=2
    l=3
    lx, ly, lz = [1,0,0,0],[0,1,1,0],[0,0,0,1]
    wx, wy, wz = [0,1,1,0],[1,0,0,1],[0,0,0,0]
    hx, hy, hz = [0,0,0,1],[0,0,0,0],[1,1,1,0]
    pos_x = (lx[i]*itemInfo[i][0] + wx[i]*itemInfo[i][1] + hx[i]*itemInfo[i][2])
    pos_y = (ly[i]*itemInfo[i][0] + wy[i]*itemInfo[i][1] + hy[i]*itemInfo[i][2])
    pos_z = (lz[i]*itemInfo[i][0] + wz[i]*itemInfo[i][1] + hz[i]*itemInfo[i][2])
    ordx = [10,0,3,11]
    ordy = [12,0,4,0]
    ordz = [4,5,1,0]
    # draw cuboids in the top left and bottom right corners, and a link between them
    cube1 = ((ordx[i]<=x)& (x<ordx[i]+pos_x)) & ((ordy[i]<=y)& (y<ordy[i]+pos_y)) & ((ordz[i]<=z)& (z<ordz[i]+pos_z))
    
    pos_x = (lx[j]*itemInfo[j][0] + wx[j]*itemInfo[j][1] + hx[j]*itemInfo[j][2])
    pos_y = (ly[j]*itemInfo[j][0] + wy[j]*itemInfo[i][1] + hy[j]*itemInfo[j][2])
    pos_z = (lz[j]*itemInfo[j][0] + wz[j]*itemInfo[j][1] + hz[j]*itemInfo[j][2])
    cube2 = ((ordx[j]<=x)& (x<ordx[j]+pos_x)) & ((ordy[j]<=y)& (y<ordy[j]+pos_y)) & ((ordz[j]<=z)& (z<ordz[j]+pos_z))
    #cube2 = ((ordx[j]<=x)& (x<ordx[j]+itemInfo[j][0])) & ((ordy[j]<=y)& (y<ordy[j]+itemInfo[j][1])) & ((ordz[j]<=z)& (z<ordz[j]+itemInfo[j][2]))
    #cube3 = ((ordx[k]<=x)& (x<ordx[k]+itemInfo[k][0])) & ((ordy[k]<=y)& (y<ordy[k]+itemInfo[k][1])) & ((ordz[k]<=z)& (z<ordz[k]+itemInfo[k][2]))
    #cube4 = ((ordx[l]<=x)& (x<ordx[l]+itemInfo[l][0])) & ((ordy[l]<=y)& (y<ordy[l]+itemInfo[l][1])) & ((ordz[l]<=z)& (z<ordz[l]+itemInfo[l][2]))

    pos_x = (lx[k]*itemInfo[k][0] + wx[k]*itemInfo[k][1] + hx[k]*itemInfo[k][2])
    pos_y = (ly[k]*itemInfo[k][0] + wy[k]*itemInfo[k][1] + hy[k]*itemInfo[k][2])
    pos_z = (lz[k]*itemInfo[k][0] + wz[k]*itemInfo[k][1] + hz[k]*itemInfo[k][2])
    cube3 = ((ordx[k]<=x)& (x<ordx[k]+pos_x)) & ((ordy[k]<=y)& (y<ordy[k]+pos_y)) & ((ordz[k]<=z)& (z<ordz[k]+pos_z))
    
    pos_x = (lx[l]*itemInfo[l][0] + wx[l]*itemInfo[l][1] + hx[l]*itemInfo[l][2])
    pos_y = (ly[l]*itemInfo[l][0] + wy[l]*itemInfo[l][1] + hy[l]*itemInfo[l][2])
    pos_z = (lz[l]*itemInfo[l][0] + wz[l]*itemInfo[l][1] + hz[l]*itemInfo[l][2])
    cube4 = ((ordx[l]<=x)& (x<ordx[l]+pos_x)) & ((ordy[l]<=y)& (y<ordy[l]+pos_y)) & ((ordz[l]<=z)& (z<ordz[l]+pos_z))
    
    # combine the objects into a single boolean array
    voxels = cube1 | cube2 | cube3 #| cube4
    
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    #colors[link] = 'red'
    colors[cube1] = 'blue'
    colors[cube2] = 'green'
    colors[cube3] = 'red'
    #colors[cube4] = 'orange'
    
    # and plot everything
    fig = plt.figure(figsize=(12,10))
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height')
    #ax.view_init(-90, 0)
    plt.show()