import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy.interpolate import BSpline
import scipy.sparse as sparse
from scipy.sparse import linalg as spla

###################################################################################### core class
class BasisFunctionExpansion_1D:

    def __init__(self,itv=None,num_basis=10):
        self.itv = itv
        self.num_basis = num_basis
        self.coef = np.zeros(num_basis)

    def set_coef(self,coef):
        self.coef = coef
        return self

    def Matrix_BasisFunction(self,x):
        pass

    def d_Matrix_BasisFunction(self,x):
        pass

    def set_x(self,x):
        [st,en] = self.itv
        self.x = x
        self.A = self.Matrix_BasisFunction(x)
        self.A_t = self.A.transpose()
        self.A_sp = sparse.csc_matrix(self.A)
        self.A_t_sp = sparse.csc_matrix(self.A_t)
        bin_edge = np.hstack([st,(x[:-1]+x[1:])/2,en])
        self.weight = bin_edge[1:] - bin_edge[:-1]
        return self

    def get_y(self):
        pass

    def get_dy(self):
        pass

    def get_y_at(self,x):
        pass

    def get_int(self):
        weight = self.weight
        y = self.get_y()
        Int = weight.dot(y)
        return Int

    def get_dint(self):
        weight = self.weight
        dy = self.get_dy()
        dInt = weight.dot(dy)
        return dInt

    def set_V(self,V):
        self.V = V
        return self

    def set_bayes(self):
        x = self.x
        d_A = self.d_Matrix_BasisFunction(x)
        W = sparse.csc_matrix(d_A.transpose().dot(d_A))
        self.W = W
        return self

    def LGH(self):
        x = self.coef; V = self.V; W = self.W;
        P = W/V
        P[0,0] += 1e-3
        log_const = logdet_sp(P)/2 - P.shape[0]*np.log(2*np.pi)/2
        L =  log_const - x.dot(P.dot(x))/2.0
        G = - P.dot(x)
        H = - P
        return [L,G,H]

    def GH_transform(self,G,H):
        A = self.A_sp; A_t = self.A_t_sp;
        G = A_t.dot(G)
        H = A_t.dot(H.dot(A))
        return [G,H]


###################################################################################### Base class
class linear_1D(BasisFunctionExpansion_1D):

    def get_y(self):
        A = self.A; coef = self.coef
        y = A.dot(coef)
        return y

    def get_dy(self):
        A = self.A;
        return A

    def get_y_at(self,x):
        coef = self.coef
        A = self.Matrix_BasisFunction(x)
        return A.dot(coef)

class loglinear_1D(BasisFunctionExpansion_1D):

    def get_y(self):
        A = self.A; coef = self.coef
        y = np.exp( A.dot(coef) )
        self.y = y
        return y

    def get_dy(self):
        A = self.A; y = self.y
        dy = y.reshape(-1,1) * A
        return dy

    def get_y_at(self,x):
        coef = self.coef
        A = self.Matrix_BasisFunction(x)
        return np.exp( A.dot(coef) )

######################################################################################  Bump function
def bump_cos(x):
    y = np.zeros_like(x)
    index = (-2<x) & (x<2)
    y[index] = ( np.cos( np.pi*x[index]/2 ) + 1 )/4
    return y

def bump_cbs(x):
    y = np.zeros_like(x)
    index = (-2<x) & (x<2)
    y[index] = BSpline.basis_element([-2,-1,0,1,2],extrapolate=False)(x[index])
    return y

def d_bump_cbs(x):
    dy = np.zeros_like(x)
    index = (-2<x) & (x<2)
    dy[index] =BSpline.basis_element([-2,-1,0,1,2],extrapolate=False).derivative()(x[index])
    return dy

def bump_plinear(x):
    y = np.interp(x,[-1,0,1],[0,1,0])
    return y

def bump_pconst(x):
    y = np.interp(x,[0,0,1,1],[0,1,1,0])
    return y

######################################################################################

########### cosine bump function
class linear_COS(linear_1D):

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-3)
        return np.vstack([ bump_cos( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

class loglinear_COS(loglinear_1D):

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-3)
        return np.vstack([ bump_cos( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

########## Cubic B-spline bump function
class linear_CBS(linear_1D):

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-3)
        return np.vstack([ bump_cbs( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

    def d_Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-3)
        return np.vstack([ d_bump_cbs( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

class loglinear_CBS(loglinear_1D):

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-3)
        return np.vstack([ bump_cbs( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

########## piecewise linear function
class plinear(linear_1D):

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-1)
        return np.vstack([ bump_plinear( (x-st-i*w)/w ) for i in range(m) ]).transpose()
    
########## piecewise constant function
class pconst(linear_1D):

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/m
        return np.vstack([ bump_pconst( (x-st-i*w)/w ) for i in range(m) ]).transpose()
    
    def get_int(self):
        [st,en] = self.itv; m = self.num_basis; 
        w = (en-st)/m
        Int = self.coef.sum()*w
        return Int

    def get_dint(self):
        [st,en] = self.itv; m = self.num_basis; 
        w = (en-st)/m
        return np.ones(m)*w
    
########## state-space model
class linear_SSM(linear_1D):
    
    def get_y(self):
        return self.coef
    
    def set_bayes(self,order=1):
        
        n = self.num_basis
        
        if order == 1:
            d0 = np.hstack((1,2*np.ones(n-2),1))
            d1 = -1*np.ones(n-1)
            data = [d1,d0,d1]
            diags = [-1,0,1]
            #rank_W = n-1
        elif order == 2:        
            d0 = np.hstack(([1,5],6*np.ones(n-4),[5,1]))
            d1 = np.hstack((-2,-4*np.ones(n-3),-2))
            d2 = np.ones(n-2)
            data = np.array([d2,d1,d0,d1,d2])
            diags = np.arange(-2,3)
            #rank_W = n-2
        
        W = sparse.diags(data,diags,shape=(n,n),format='csc')
        self.W = W
        return self

    def GH_transform(self,G,H):
        return [G,H]

###################################################################### basic routines
def logdet_sp(P):
    LU = spla.splu(P)
    logdet = np.log(np.abs(LU.U.diagonal())).sum() + np.log(np.abs(LU.L.diagonal())).sum()
    #logdet = np.linalg.slogdet(P.toarray())[1]
    return logdet
