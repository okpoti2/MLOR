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

import numpy as np
from viewResults import viewResults
import multiprocessing

class validateSequence:
    
    def __init__(self,itemInfo, boxInfo):
        self.itemInfo = itemInfo
        self.boxInfo = boxInfo
        self.numBox = len(boxInfo)
        self.numItem = len(itemInfo)
        self.s_out = {}
        self.x_out, self.y_out, self.z_out = [],[],[]
        self.n_out = [1]*self.numBox
        self.a_out, self.b_out, self.c_out, self.d_out = {},{},{},{}
        self.e_out, self.f_out, self.g_out = {},{},{}
        self.lx_out, self.ly_out, self.lz_out = [],[],[]
        self.wx_out, self.wy_out, self.wz_out = [],[],[]
        self.hx_out, self.hy_out, self.hz_out = [],[],[]
        
    def plotResults(self,seq):
        vr = viewResults(self.itemInfo, self.boxInfo,self.s_out)
        vr.setSpatialCoordinates(self.x_out, self.y_out, self.z_out)
        vr.setLengthAlign(self.lx_out, self.ly_out, self.lz_out)
        vr.setWidthAlign(self.wx_out, self.wy_out, self.wz_out)
        vr.setHeightAlign(self.hx_out, self.hy_out, self.hz_out)
        itemsList = []
        
        for j in range(1):#self.numBox):
            for k in range(self.numItem):
                i = seq[k]-1
                if (i,j) in self.s_out and self.s_out[i,j]>0:
                    itemsList.append(i)
        #itemsList = [k[0] for k in self.s_out if self.s_out[k]>0]
        print(itemsList)
        vr.plotOneResult(0, itemsList)
    
    def binAssignment(self):
        p,q,r = [],[],[] # item length, width and height
        L,W,H = [],[],[] # box length, width and height
        
        for i in range(self.numItem):
            p.append(self.itemInfo[i][0])
            q.append(self.itemInfo[i][1])
            r.append(self.itemInfo[i][2])
        
        for i in range(self.numBox):
            L.append(self.boxInfo[i][0])
            W.append(self.boxInfo[i][1])
            H.append(self.boxInfo[i][2])
        
        
        bin_dict = {i:L[i]*W[i]*H[i] for i in range(self.numBox)}
        bin_dict_raw = {i:L[i]*W[i]*H[i] for i in range(self.numBox)}
        bin_assign = [0]
        bin_dict[0] -= p[0]*q[0]*r[0]
        bin_dict_raw[0] -= p[0]*q[0]*r[0]
        self.s_out[0,0] = 1
        itemVolumes = np.array([self.itemInfo[i][0]*self.itemInfo[i][1]*self.itemInfo[i][2] for i in range(self.numItem)])
        eps = np.average(itemVolumes)
        for i in range(1,len(self.itemInfo)):
            arr = np.array(list(bin_dict.values()))
            
            #eps = np.average(self.itemInfo)
            idx = np.where(arr - ((p[i]*q[i]*r[i])+eps) < 0, np.inf, arr - ((p[i]*q[i]*r[i])+eps)).argmin()
            bin_dict[idx] -= (p[i]*q[i]*r[i])
            bin_dict_raw[idx] -= (p[i]*q[i]*r[i])
            bin_assign.append(idx)
            self.s_out[i,idx] = 1
        print('Pre-optimization bins = ',len(set(bin_assign)))
        totalVolUsed = np.array([bin_dict_raw[i] for i in set(bin_assign)])
        totalVolUsed = np.sum(totalVolUsed)
        
        return [len(set(bin_assign)),totalVolUsed]
        
            
    def binAssignmentExpanded(self,seq=None):
        num_features = 3
        p,q,r = [],[],[] # item length, width and height
        L,W,H = [],[],[] # box length, width and height
        
        def rotateBox(costMatrix): #helper function to determine orientation
            rowlen = costMatrix.shape[0]
            rotation = np.zeros((rowlen,rowlen))
            rows = [i for i in range(rowlen)]
            cols = rows.copy()
            val =0
            while rowlen > 0:
                val += np.amin(costMatrix)
                flat_index = np.argmin(costMatrix)
                row_index = flat_index//costMatrix.shape[0]
                col_index = flat_index%costMatrix.shape[0]
                rotation[rows[row_index],cols[col_index]] = 1
                rowlen -=1
                costMatrix = np.delete(costMatrix,row_index,0)
                costMatrix = np.delete(costMatrix,col_index,1)
                rows.pop(row_index)
                cols.pop(col_index)
            return rotation,val
        
        
        def isOrientationFeasible(binDim,prevItemCoord,currentItemDim):
            ''' determines if the current item really fits the available space
                considering all possible rotations and items already in bin'''
            isFeasible = False
            relativeCost = np.array([prevItemCoord + currentItemDim[i] for i in range(len(prevItemCoord))])
            rotation, cost = rotateBox(relativeCost)
            costFeasibility = np.where(rotation == 1, relativeCost, np.inf)
            costFeasibilityB = costFeasibility.copy()
            costFeasibility = np.array([np.min(costFeasibility[i]) for i in range(costFeasibility.shape[0])])
            costFeasibility = np.where(costFeasibility <= binDim, 1, 0)
            checkB = 0
            for i in range(costFeasibilityB.shape[0]):
                for j in range(len(costFeasibilityB[i])):
                    if rotation[i][j]==1 and costFeasibilityB[i][j]<=binDim[j]:
                        checkB+=1
            #print(checkB)           
            #print(rotation)
            if sum(costFeasibility)==len(binDim) or checkB>=len(binDim)-1:
                isFeasible = True
            
            return isFeasible
                
        
        def assignOrientationVariables(item,rotation):
            self.lx_out[item], self.ly_out[item], self.lz_out[item] = rotation[0,0], rotation[0,1], rotation[0,2]
            self.wx_out[item], self.wy_out[item], self.wz_out[item] = rotation[1,0], rotation[1,1], rotation[1,2]
            self.hx_out[item], self.hy_out[item], self.hz_out[item] = rotation[2,0], rotation[2,1], rotation[2,2]
        
        
        def assignPositionCoordinates(item,coord):
            self.x_out[item], self.y_out[item], self.z_out[item] = coord[0], coord[1], coord[2]
        
        for i in range(self.numItem):
            p.append(self.itemInfo[i][0])
            q.append(self.itemInfo[i][1])
            r.append(self.itemInfo[i][2])
        
        for i in range(self.numBox):
            L.append(self.boxInfo[i][0])
            W.append(self.boxInfo[i][1])
            H.append(self.boxInfo[i][2])
            
        # Initialize coordinate and orientation varables
        self.x_out, self.y_out, self.z_out = [0]*self.numItem,[0]*self.numItem,[0]*self.numItem
        self.lx_out, self.ly_out, self.lz_out = [0]*self.numItem,[0]*self.numItem,[0]*self.numItem
        self.wx_out, self.wy_out, self.wz_out = [0]*self.numItem,[0]*self.numItem,[0]*self.numItem
        self.hx_out, self.hy_out, self.hz_out = [0]*self.numItem,[0]*self.numItem,[0]*self.numItem   
        
        bin_volume = np.array([L[i]*W[i]*H[i] for i in range(self.numBox)]) # pre-determined volume of each bin
        item_volume = np.array([p[i]*q[i]*r[i] for i in range(self.numItem)]) #pre-determined volume of each item
        position_coord = np.array([[0]*num_features for i in range(self.numItem)]) # x,y,z values to be determined for each item
        binItemAssigned = {i:[] for i in range(self.numBox)} # items that have been assigned to the bin
        binsWithAssignedItems = []
        binDim = np.array([[L[i],W[i],H[i]] for i in range(self.numBox)]) # pre-determined volume of each bin
        itemDim = np.array([[p[i],q[i],r[i]] for i in range(self.numItem)])
        
        f = seq[0]-1 # for 0 indexing
        bin_volume[0] -= item_volume[f] # utilized volume tracker
        self.s_out[f,0] = 1
        binsWithAssignedItems.append(0) #zero start indexing
        binItemAssigned[0].append(f) # bin 1 assigned item 1
        eps = np.average(item_volume)
        
        #determine rotation for first item
        rotationCost = np.array([self.boxInfo[0][i] - itemDim[f] for i in range(num_features)])
        rotationCost = np.where(rotationCost < 0, np.inf, rotationCost)
        rotation,cost = rotateBox(rotationCost)
        assignOrientationVariables(f,rotation)

        #determine coordinates for first item
        positionCost = rotationCost.copy()
        positionCost = np.where(positionCost < 0, np.inf, positionCost)
        coord = [np.min(positionCost[i]) for i in range(positionCost.shape[0])]
        position_coord[f] = np.array(coord)#@rotation
        assignPositionCoordinates(f,position_coord[f])
        
        #Determine rotations and coordinates for subsequent items
        for k in range(1,4):#self.numItem):
            i = seq[k] - 1
            # Determine a feasible bin by checking available volume - level 1
            avail_volumes = bin_volume#np.array(list(bin_volume.values()))
            feas_bin = np.where(avail_volumes - item_volume[i] - eps < 0, np.inf, avail_volumes - item_volume[i]- eps).argmin()
            
            # Determine item rotation feasiblity with respect to already added items - level 2
            itemFitsBin = False
            for item in binItemAssigned[feas_bin]:
                totalCost = np.inf
                if isOrientationFeasible(binDim[feas_bin],position_coord[item],itemDim[i]):
                    relativeCostA = np.array([itemDim[i][j] - position_coord[item] for j in range(len(itemDim[i]))])
                    relativeCostA = np.where(relativeCostA < 0, np.inf, relativeCostA)
                    relativeCostB = np.array([itemDim[i][j] + position_coord[item] for j in range(len(itemDim[i]))])
                    for row in range(relativeCostB.shape[0]):
                        for col in range(relativeCostB.shape[1]):
                            if relativeCostB[row][col] > binDim[feas_bin][row]:
                                relativeCostB[row][col] = np.inf
                    relativeCost = np.min([relativeCostA, relativeCostB])
                    print(relativeCost)
                    rotation, cost = rotateBox(relativeCost)
                    if cost < totalCost:
                        assignOrientationVariables(i,rotation)
                        itemFitsBin = True
                    
            
            if itemFitsBin==False:
                feas_bin = binsWithAssignedItems[-1] + 1
                relativeCost = np.array([self.boxInfo[feas_bin][j] - itemDim[i] for j in range(len(itemDim[i]))])
                relativeCost = np.where(relativeCost < 0, np.inf, relativeCost)
                rotation,cost = rotateBox(relativeCost)
                assignOrientationVariables(i,rotation)

            #print(feas_bin,'-----')
            
            # ----Determine the actual x,y,z coordinate values for the current item------------------------
            # This will be done looking at each item in the assigned feasible bin
            xPos_RL,yPos_FB,zPos_UD = [],[],[] # x for right/left, y for front/back, z for up/down
            itemFitsBin = False
            rotatedItemDim = itemDim[i]@rotation
            print(rotatedItemDim,'for ',i,'-------rotated dimension')
            if len(binItemAssigned[feas_bin])>0:
                for item in binItemAssigned[feas_bin]:
                    a = rotatedItemDim[0]
                    b = rotatedItemDim[1]
                    c = rotatedItemDim[2]
                    xPos_RL.append([itemDim[item][0]+position_coord[item][0] - a , position_coord[item][0] - a , position_coord[item][0] + itemDim[item][0], self.boxInfo[feas_bin][0] - a])
                    yPos_FB.append([itemDim[item][1]+position_coord[item][1] - b , position_coord[item][1] - b , position_coord[item][1] + itemDim[item][1], self.boxInfo[feas_bin][1] - b])
                    zPos_UD.append([itemDim[item][2]+position_coord[item][2] - c ,position_coord[item][2] - c , position_coord[item][2] + itemDim[item][2], self.boxInfo[feas_bin][2] - c])
                
                #convert to a numpy array for easy and fast manipulation
                xPos_RL = np.array(xPos_RL)
                yPos_FB = np.array(yPos_FB)
                zPos_UD = np.array(zPos_UD)
                #print(zPos_UD)
                #Elimiate instances of negative values
                xPos_RL = np.where(xPos_RL < 0, np.inf, xPos_RL)
                yPos_FB = np.where(yPos_FB < 0, np.inf, yPos_FB)
                zPos_UD = np.where(zPos_UD < 0, np.inf, zPos_UD)
                
                x = np.min(xPos_RL)
                y = np.min(yPos_FB)
                z = np.min(zPos_UD)
                
                if x >=np.inf:
                    x = 0
                if y >=np.inf:
                    y = 0
                if z >=np.inf:
                    z = 0
                coord = np.array([x,y,z])#@rotation
                #assignPositionCoordinates(i,coord)
            #else:
                #coord = [0,0,0]
            x = np.max
            print('position:',coord)
            assignPositionCoordinates(i,coord)
            if feas_bin not in binsWithAssignedItems:
                binsWithAssignedItems.append(feas_bin)
            #print(relativeCost)
            bin_volume[feas_bin] -= item_volume[i] # utilized volume
            binItemAssigned[feas_bin].append(i)
            self.s_out[i,feas_bin] = 1
            
                
            
            

        print('Pre-optimization bins = ',len(binsWithAssignedItems))
        self.plotResults(seq)
    
    
    
    def evaluate(self,output=None, seq=None):
        from docplex.mp.model import Model
        #self.binAssignment()
        #self.binAssignmentExpanded()
        
        p,q,r = [],[],[] # item length, width and height
        L,W,H = [],[],[] # box length, width and height
        M = 9999999999999
        
        for i in range(self.numItem):
            p.append(self.itemInfo[i][0])
            q.append(self.itemInfo[i][1])
            r.append(self.itemInfo[i][2])
        
        for i in range(self.numBox):
            L.append(self.boxInfo[i][0])
            W.append(self.boxInfo[i][1])
            H.append(self.boxInfo[i][2])
        
        # iterate over new sequence and place item one-by-one
        originalBoxNum = self.numBox
        interval = 5
        for yy in range(5,len(self.itemInfo),interval):
            self.numItem = yy+1
            if yy < originalBoxNum:
                self.numBox = yy
            else:
                self.numBox = originalBoxNum
            
            bpp = Model(name='3D Bin Packing Problem') # Model object
            #bpp.set_time_limit(60)
            bpp.context.cplex_parameters.threads = multiprocessing.cpu_count()+1
            #----------------------------------------------------------------------
            # Decision variable declaration
            # ---------------------------------------------------------------------
            A = [(i,j) for i in range(self.numItem) for j in range(self.numBox)]
            s = bpp.binary_var_dict(A,name="s") #1 if item i is assigned to box j, 0 otherwise
            
            n = bpp.binary_var_list(range(self.numBox),name="n") #1 if box j is used, 0 otherwise
            
            #position coordinates variables
            x = bpp.continuous_var_list(range(self.numItem),lb=0,name="x") # x-axis position of item
            y = bpp.continuous_var_list(range(self.numItem),lb=0,name="y") # y-axis position of item
            z = bpp.continuous_var_list(range(self.numItem),lb=0,name="z") # z-axis position of item
            
            # packing of boxes variables
            lx = bpp.binary_var_list(range(self.numItem),name="lx") # length of item i is parallel to x-axis
            ly = bpp.binary_var_list(range(self.numItem),name="ly") # length of item i is parallel to y-axis
            lz = bpp.binary_var_list(range(self.numItem),name="lz") # length of item i is parallel to z-axis
            wx = bpp.binary_var_list(range(self.numItem),name="wx") # width of item i is parallel to x-axis
            wy = bpp.binary_var_list(range(self.numItem),name="wy") # width  of item i is parallel to y-axis
            wz = bpp.binary_var_list(range(self.numItem),name="wz") # width  of item i is parallel to z-axis
            hx = bpp.binary_var_list(range(self.numItem),name="hx") # height of item i is parallel to x-axis
            hy = bpp.binary_var_list(range(self.numItem),name="hy") # height of item i is parallel to y-axis
            hz = bpp.binary_var_list(range(self.numItem),name="hz") # height of item i is parallel to z-axis
            
            # relative packing variables
            B = [(i,k) for i in range(self.numItem) for k in range(self.numItem)]
            a = bpp.binary_var_dict(B,name="a") # 1 if item i is on the left side of item k, else 0
            b = bpp.binary_var_dict(B,name="b") # 1 if item i is on the right side of item k, else 0
            c = bpp.binary_var_dict(B,name="c") # 1 if item i is behind item k, else 0
            d = bpp.binary_var_dict(B,name="d") # 1 if item i is in front of item k, else 0
            e = bpp.binary_var_dict(B,name="e") # 1 if item i is below item k, else 0
            f = bpp.binary_var_dict(B,name="f") # 1 if item i is above item k, else 0
            g = bpp.binary_var_dict(B,name="g") # 1 if items i and item k are assigned together, else 0
            
            
            #----------------------------------------------------------------------
            # Objective function
            # ---------------------------------------------------------------------
            boxVolume = 0
            for j in range(self.numBox):
                boxVolume += L[j] * W[j] * H[j] * n[j]
            
            itemsVolume = 0
            for i in range(self.numItem):
                itemsVolume += p[i] * q[i] * r[i]
            obj = boxVolume - itemsVolume
            bpp.minimize(obj)
            
            
            #----------------------------------------------------------------------
            # Constraints declaration
            # ---------------------------------------------------------------------
            #constraint 1
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = x[i] + p[i]*lx[i] + q[i]*wx[i] + r[i]*hx[i] 
                    rhs = x[k] + M*(1-a[i,k])
                    const1 = lhs<=rhs
                    bpp.add_constraint(const1)
            
            #constraint 2
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = x[k] + p[k]*lx[k] + q[k]*wx[k] + r[k]*hx[k] 
                    rhs = x[i] + M*(1-b[i,k])
                    const2 = lhs<=rhs
                    bpp.add_constraint(const2)
                    
            #constraint 3
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = y[i] + p[i]*ly[i] + q[i]*wy[i] + r[i]*hy[i] 
                    rhs = y[k] + M*(1-c[i,k])
                    const3 = lhs<=rhs
                    bpp.add_constraint(const3)
                    
            #constraint 4
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = y[k] + p[k]*ly[k] + q[k]*wy[k] + r[k]*hy[k] 
                    rhs = y[i] + M*(1-d[i,k])
                    const4 = lhs<=rhs
                    bpp.add_constraint(const4)
                    
            #constraint 5
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = z[i] + p[i]*lz[i] + q[i]*wz[i] + r[i]*hz[i] 
                    rhs = z[k] + M*(1-e[i,k])
                    const5 = lhs<=rhs
                    bpp.add_constraint(const5)
                    
            #constraint 6
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = z[k] + p[k]*lz[k] + q[k]*wz[k] + r[k]*hz[k] 
                    rhs = z[i] + M*(1-f[i,k])
                    const6 = lhs<=rhs
                    bpp.add_constraint(const6)
            
            #constraint 7
            for i in range(self.numItem):
                lhs = lx[i] + ly[i] + lz[i]
                const7 = lhs==1
                bpp.add_constraint(const7)
            
            #constraint 8
            for i in range(self.numItem):
                lhs = wx[i] + wy[i] + wz[i]
                const8 = lhs==1
                bpp.add_constraint(const8)
                
            #constraint 9
            for i in range(self.numItem):
                lhs = hx[i] + hy[i] + hz[i]
                const9 = lhs==1
                bpp.add_constraint(const9)
                
            #constraint 10
            for i in range(self.numItem):
                lhs = lx[i] + wx[i] + hx[i]
                const10 = lhs==1
                bpp.add_constraint(const10)
                
            #constraint 11
            for i in range(self.numItem):
                lhs = ly[i] + wy[i] + hy[i]
                const11 = lhs==1
                bpp.add_constraint(const11)
                
            #constraint 12
            for i in range(self.numItem):
                lhs = lz[i] + wz[i] + hz[i]
                const12 = lhs==1
                bpp.add_constraint(const12)
            
            #constraint 13
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    lhs = a[i,k] + b[i,k] + c[i,k] + d[i,k] + e[i,k] + f[i,k]
                    const13 = lhs >= g[i,k]
                    bpp.add_constraint(const13)
            
            #constraint 14
            for i in range(self.numItem):
                totalBins = 0
                for j in range(self.numBox):
                    totalBins+=s[i,j]
                const14 = totalBins == 1
                bpp.add_constraint(const14)
            
            #constraint 15
            for j in range(self.numBox):
                totalAssignedBins = 0
                for i in range(self.numItem):
                    totalAssignedBins+=s[i,j]
                const15 = totalAssignedBins <= M * n[j]
                bpp.add_constraint(const15)
                
            #constraint 16
            for i in range(self.numItem):
                for k in range(i+1,self.numItem):
                    for j in range(self.numBox):
                        lhs = s[i,j] + s[k,j]
                        const16 = lhs <= g[i,k] + 1
                        bpp.add_constraint(const16)
            
            #constraint 17
            for i in range(self.numItem):
                for j in range(self.numBox):
                    lhs = x[i] + p[i]*lx[i] + q[i]*wx[i] + r[i]*hx[i] 
                    rhs = L[j] + M*(1 - s[i,j])
                    const17 = lhs<=rhs
                    bpp.add_constraint(const17)
            
            #constraint 18
            for i in range(self.numItem):
                for j in range(self.numBox):
                    lhs = y[i] + p[i]*ly[i] + q[i]*wy[i] + r[i]*hy[i] 
                    rhs = W[j] + M*(1 - s[i,j])
                    const18 = lhs<=rhs
                    bpp.add_constraint(const18)
                    
            #constraint 19
            for i in range(self.numItem):
                for j in range(self.numBox):
                    lhs = z[i] + p[i]*lz[i] + q[i]*wz[i] + r[i]*hz[i] 
                    rhs = H[j] + M*(1 - s[i,j])
                    const19 = lhs<=rhs
                    bpp.add_constraint(const19)
            
            bpp.add_constraint(x[0]==0)
            bpp.add_constraint(y[0]==0)
            bpp.add_constraint(z[0]==0)
            #---------------constraints to ensure continuity------------------
            if len(self.s_out) > 0:
                temp_bins = []
                for i in range(self.numItem-interval):
                    for j in range(self.numBox):
                        #if i < self.numItem-1:
                        if (i,j) in self.s_out.keys():
                            bpp.add_constraint(s[i,j] == self.s_out[i,j])
                            if j not in temp_bins and self.s_out[i,j]==1:
                                #bpp.add_constraint(n[j]==self.n_out[j])
                                temp_bins.append(j)
                        #else:
                            #bpp.add_constraint(s[i,j] == 0)
                                
            if len(self.x_out) > 0:
                for i in range(self.numItem):
                    if i < self.numItem-interval:
                        bpp.add_constraint(x[i] == self.x_out[i])
                        bpp.add_constraint(y[i] == self.y_out[i])
                        bpp.add_constraint(z[i] == self.z_out[i])
                        
                        bpp.add_constraint(lx[i] == self.lx_out[i])
                        bpp.add_constraint(ly[i] == self.ly_out[i])
                        bpp.add_constraint(lz[i] == self.lz_out[i])
                        bpp.add_constraint(wx[i] == self.wx_out[i])
                        bpp.add_constraint(wy[i] == self.wy_out[i])
                        bpp.add_constraint(wz[i] == self.wz_out[i])
                        bpp.add_constraint(hx[i] == self.hx_out[i])
                        bpp.add_constraint(hy[i] == self.hy_out[i])
                        bpp.add_constraint(hz[i] == self.hz_out[i])
                        
            if len(self.a_out)>0:
                for i in range(self.numItem-interval):
                    for k in range(self.numItem-interval):
                        #k = self.numItem-1
                        if i < self.numItem-1 and k < self.numItem-1:
                            bpp.add_constraint(a[i,k] == self.a_out[i,k])
                            bpp.add_constraint(b[i,k] == self.b_out[i,k])
                            bpp.add_constraint(c[i,k] == self.c_out[i,k])
                            bpp.add_constraint(d[i,k] == self.d_out[i,k])
                            bpp.add_constraint(e[i,k] == self.e_out[i,k])
                            bpp.add_constraint(f[i,k] == self.f_out[i,k])
                            bpp.add_constraint(g[i,k] == self.g_out[i,k])
                            
            bpps = bpp.solve()
            if bpps:
                print(bpps._solve_status)
                self.s_out = bpps.get_value_dict(s)
                self.x_out = bpps.get_values(x)
                self.y_out = bpps.get_values(y)
                self.z_out = bpps.get_values(z)
                
                self.a_out = bpps.get_value_dict(a)
                self.b_out = bpps.get_value_dict(b)
                self.c_out = bpps.get_value_dict(c)
                self.d_out = bpps.get_value_dict(d)
                self.e_out = bpps.get_value_dict(e)
                self.f_out = bpps.get_value_dict(f)
                self.g_out = bpps.get_value_dict(g)
                
                self.lx_out = bpps.get_values(lx)
                self.ly_out = bpps.get_values(ly)
                self.lz_out = bpps.get_values(lz)
                self.wx_out = bpps.get_values(wx)
                self.wy_out = bpps.get_values(wy)
                self.wz_out = bpps.get_values(wz)
                self.hx_out = bpps.get_values(hx)
                self.hy_out = bpps.get_values(hy)
                self.hz_out = bpps.get_values(hz)
                
                #plot the result and save
                #self.plotResults(seq)
            else:
                return ['Not solved','Not solved']
            
            numBinsUsed = sum(bpps.get_values(n))
            volumeUtilized = bpp.objective_value
            if yy < len(self.itemInfo)-1:
                del bpp
            
        
        if output is not None:
            print('Total volume used is {}'.format(volumeUtilized))
            print('Total assigned bins is {} \n'.format(numBinsUsed))
            #plot the result and save
            self.plotResults(seq)
            if seq is not None:
                print('Item-to-bin assignment is as follows:')
                for j in range(self.numBox):
                    out = 'Bin '+str(j+1)+': {'
                    for i in range(self.numItem):
                        if self.s_out[i,j]>0:
                            out+=str(seq[i]) + ' '
                    out +='}'
                    print(out)
        return [numBinsUsed, volumeUtilized]
        