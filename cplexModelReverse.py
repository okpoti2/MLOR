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

class mathModel:
    
    def __init__(self,itemFile, boxFile):
        self.itemFilePath = itemFile
        self.boxFilePath  = boxFile

        self.itemInfo = np.loadtxt(self.itemFilePath,delimiter=',')
        self.numItem = len(self.itemInfo)
        
        self.boxInfo = np.loadtxt(self.boxFilePath,delimiter=',')
        self.numBox = len(self.boxInfo)
    
    def get_itemInfo(self):
        ''' returns the data containing length, width
        and height information on items'''
        return self.itemInfo
    
    def get_boxInfo(self):
        ''' returns the data containing length, width
        and height information on boxes'''
        return self.boxInfo
    
    #Solve model
    def solveModel(self):
        from docplex.mp.model import Model
        p,q,r = [],[],[] # item length, width and height
        L,W,H = [],[],[] # box length, width and height
        M = 999999999
        
        for i in range(self.numItem):
            p.append(self.itemInfo[i][0])
            q.append(self.itemInfo[i][1])
            r.append(self.itemInfo[i][2])
        
        for i in range(self.numBox):
            L.append(self.boxInfo[i][0])
            W.append(self.boxInfo[i][1])
            H.append(self.boxInfo[i][2])
        
        bpp = Model(name='3D Bin Packing Problem') # Model object
        
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
                    const16 = lhs <= g[i,k] +1
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
        
        # extra constraints
        #bpp.add_constraint(x[0]==0)
        #bpp.add_constraint(y[0]==0)
        #bpp.add_constraint(z[0]==0)
        
        bpps = bpp.solve()
        print(bpps.get_value_dict(s))
        print('X-axis {}'.format(bpps.get_values(x)))
        print('Y-axis {}'.format(bpps.get_values(y)))
        print('Z-axis {}'.format(bpps.get_values(z)))
        print('Objective value is {}'.format(bpp.objective_value))
        print('Assigned bins are {}'.format(bpps.get_values(n)))

if __name__ == '__main__':
    
    #Read data files
    path_to_itemFile = 'data/item-chen.txt'
    path_to_boxFile = 'data/box-chen.txt'
    optModel = mathModel(path_to_itemFile,path_to_boxFile)
    optModel.solveModel()