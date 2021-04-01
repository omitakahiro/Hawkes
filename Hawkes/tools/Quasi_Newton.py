import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl

##################################
## Array labeling
##################################
class array_label:
    
    def __init__(self,para_list,para_length):
        
        self._para_list = para_list
        self._para_length = para_length
        self._length = np.array(para_length).sum()
        self._hash_table = {}
        index_ini = 0

        for para,length in zip(para_list,para_length):
            self._hash_table.update({ para: slice(index_ini,index_ini+length) if length>1 else index_ini })
            self._hash_table.update({ (para,i): index_ini+i for i in range(length) })
            index_ini += length

    ### hash table
    def idx(self,key):
        return self._hash_table[key]
    
    def _index(self,key_list):
        try:
            index = np.hstack([ np.arange(self._length,dtype='i8')[self._hash_table[key]] for key in key_list ])
        except:
            index = slice(0)
        return index

    def add_key(self,key_list,key_name):
        self._hash_table.update({ key_name: self._index(key_list) })
        return self
    
    ### I/O
    def from_dict(self,dic):
        x = np.zeros(self._length)
        for key in dic:
            try:
                x[self.idx(key)] = dic[key]
            except:
                x[self.idx(key)] = dic[key][0]
        return x
    
    def to_dict(self,ndarray):
        return { para: ndarray[self.idx(para)] for para in self._para_list }
    
##################################
## Quasi Newton
##################################
def Quasi_Newton(model,prior=[],merge=[],opt=[]):

    ## parameter setting
    para_list  = model.stg["para_list"]
    para_length = [ model.stg["para_length"][key] for key in para_list ]
    param = array_label(para_list,para_length)
    param.add_key( [ pr for pr in para_list if     model.stg["para_exp"][pr] ], "para_exp")
    param.add_key( [ pr for pr in para_list if not model.stg["para_exp"][pr] ], "para_ord")
    model.stg['para_label'] = param

    if 'para_ini' not in opt:
        para = param.from_dict(model.stg['para_ini'])
    else:
        para = param.from_dict(opt['para_ini'])

    step_Q = param.from_dict(model.stg["para_step_Q"])
    m = len(para)

    ## prior setting
    if prior:
        ##fix check
        para_fix_index = [ (prior_i["name"],prior_i["index"]) for prior_i in prior if prior_i["type"] == "f" ]
        para_fix_value = [ prior_i["mu"] for prior_i in prior if prior_i["type"] == "f" ]
        param.add_key(para_fix_index,"fix")
        para[param.idx('fix')] = para_fix_value
        prior = [ prior_i for prior_i in prior if prior_i["type"] != "f" ]
    else:
        param.add_key([],"fix")

    ## merge setting
    if merge:
        d = len(merge)
        index_merge = np.zeros(m,dtype='i8')

        for i in range(d):
            key = 'merge%d' % i
            param.add_key(merge[i],key)
            para[param.idx(key)] = para[param.idx(key)].mean()
            index_merge[param.idx(key)] = i+1

        M_merge_z = np.eye(m)[ index_merge == 0 ]
        M_merge_nz = np.vstack(  [ np.eye(m)[index_merge == i+1].sum(axis=0) for i in range(d) ] )
        M_merge = np.vstack([M_merge_z,M_merge_nz])
        M_merge_T = np.transpose(M_merge)
        m_reduced = M_merge.shape[0]

    else:
        M_merge = 1
        M_merge_T = 1
        m_reduced = m

    # calculate Likelihood and Gradient at the initial state
    [L1,G1] = Penalized_LG(model,para,prior)
    G1[param.idx("para_exp")] *= para[param.idx("para_exp")]
    G1 = np.dot(M_merge,G1)

    # main
    H = np.eye(m_reduced)
    i_loop = 0

    while 1:

        if 'print' in opt:
            print(i_loop)
            print(param.to_dict(para))
            #print(G1)
            print( "L = %.3f, norm(G) = %e\n" % (L1,np.linalg.norm(G1)) )
            #sys.exit()

        if 'stop' in opt:
            if i_loop == opt['stop']:
                break

        #break rule
        if np.linalg.norm(G1) < 1e-5 :
            break

        #calculate direction
        s = H.dot(G1);
        s_extended = np.dot(M_merge_T,s)
        gamma = 1/np.max([np.max(np.abs(s_extended)/step_Q),1])
        s = s * gamma
        s_extended = s_extended * gamma

        #move to new point
        para[param.idx("para_ord")] += s_extended[param.idx("para_ord")]
        para[param.idx("para_exp")] *= np.exp( s_extended[param.idx("para_exp")] )

        #calculate Likelihood and Gradient at the new point
        [L2,G2] = Penalized_LG(model,para,prior)
        G2[param.idx("para_exp")] *= para[param.idx("para_exp")]
        G2 = np.dot(M_merge,G2)

        #update hessian matrix
        y = (G1-G2).reshape(-1,1)
        s = s.reshape(-1,1)

        if  y.T.dot(s) > 0:
            H = H + (y.T.dot(s)+y.T.dot(H).dot(y))*(s*s.T)/(y.T.dot(s))**2 - (H.dot(y)*s.T+(s*y.T).dot(H))/(y.T.dot(s))
        else:
            H = np.eye(m_reduced)

        #update Gradients
        L1 = L2
        G1 = G2

        i_loop += 1

    ###OPTION: Estimation Error
    if 'ste' in opt:
        ste = EstimationError(model,para,prior)
    else:
        ste = []

    ###OPTION: Check map solution
    if 'check' in opt:
            Check_QN(model,para,prior)

    return [param.to_dict(para),L1,ste,np.linalg.norm(G1),i_loop]

def Check_QN(model,para,prior):
    param = model.stg['para_label']
    ste = EstimationError_approx(model,para,prior)
    ste[param.idx('fix')] = 0
    a = np.linspace(-1,1,21)
    
    for key in model.stg['para_list']:
        for index in range(model.stg['para_length'][key]):
            
            plt.figure()
            plt.title(key + '-' + str(index))

            for i in range(len(a)):
                para_tmp = para.copy()
                para_tmp[param.idx((key,index))] += a[i] * ste[param.idx((key,index))]
                L = Penalized_LG(model,para_tmp,prior)[0]
                plt.plot(para_tmp[param.idx((key,index))],L,"ko")

                if i==10:
                    plt.plot(para_tmp[param.idx((key,index))],L,"ro")

#################################
## Basic funnctions
#################################
def G_NUMERICAL(model,para):
    
    m = len(para)
    param = model.stg['para_label']
    step_diff = param.from_dict(model.stg['para_step_diff'])
    step_diff[param.idx('para_exp')] *= para[param.idx('para_exp')]
    G = np.zeros(m)
    
    for i in range(m):
        step = step_diff[i]

        """
        para_tmp = para.copy(); para_tmp[i] -= step;  L1 = model.LG(param.to_dict(para_tmp))[0]
        para_tmp = para.copy(); para_tmp[i] += step;  L2 = model.LG(param.to_dict(para_tmp))[0]
        G[i]= (L2-L1)/2/step
        """

        para_tmp = para.copy(); para_tmp[i] -= 2*step;  L1 = model.LG(param.to_dict(para_tmp))[0]
        para_tmp = para.copy(); para_tmp[i] -= 1*step;  L2 = model.LG(param.to_dict(para_tmp))[0]
        para_tmp = para.copy(); para_tmp[i] += 1*step;  L3 = model.LG(param.to_dict(para_tmp))[0]
        para_tmp = para.copy(); para_tmp[i] += 2*step;  L4 = model.LG(param.to_dict(para_tmp))[0]
        G[i]= (L1-8*L2+8*L3-L4)/12/step

    return G


def Hessian(model,para,prior):

    m = len(para)
    param = model.stg['para_label']
    step_diff = param.from_dict(model.stg['para_step_diff'])
    step_diff[param.idx('para_exp')] *= para[param.idx('para_exp')]
    H = np.zeros((m,m))
    
    for i in range(m):
        step = step_diff[i]
        para_tmp = para.copy(); para_tmp[i] -= step;  G1 = Penalized_LG(model,para_tmp,prior)[1]
        para_tmp = para.copy(); para_tmp[i] += step;  G2 = Penalized_LG(model,para_tmp,prior)[1]
        H[i] = (G2-G1)/2/step

    H[param.idx('fix')] = 0
    H[param.idx('fix'),param.idx('fix')] = -1e+20

    return H

def EstimationError(model,para,prior):
    H = Hessian(model,para,prior)
    ste = np.sqrt(np.diag(np.linalg.inv(-H)))
    return ste

def EstimationError_approx(model,para,prior):
    H = Hessian(model,para,prior)
    ste = 1.0/np.sqrt(np.diag(-H))
    return ste


def Penalized_LG(model,para,prior,only_L=False):
    
    param = model.stg['para_label']

    [L,G] = model.LG(param.to_dict(para),only_L)

    if isinstance(G,str):
        G = G_NUMERICAL(model,para)
    else:
        G = param.from_dict(G)

    ## fix
    if not only_L:
        G[param.idx("fix")] = 0

    ## prior
    if prior:

        for prior_i in prior:
            para_key = prior_i["name"]
            para_index = prior_i["index"]
            prior_type = prior_i["type"]
            mu = prior_i["mu"]
            sigma = prior_i["sigma"]
            index = param.idx((para_key,para_index))
            x = para[index]

            if prior_type == 'n': #prior: normal distribution
                L  += - np.log(2*np.pi*sigma**2)/2 - (x-mu)**2/2/sigma**2
                if not only_L:
                    G[index] += - (x-mu)/sigma**2
            elif prior_type ==  'ln': #prior: log-normal distribution
                L  += - np.log(2*np.pi*sigma**2)/2 - np.log(x) - (np.log(x)-mu)**2/2/sigma**2
                if not only_L:
                    G[index] += - 1/x - (np.log(x)-mu)/sigma**2/x
            elif prior_type == "b": #prior: barrier function
                L  += - mu/x
                if not only_L:
                    G[index] += mu/x**2
            elif prior_type == "b2": #prior: barrier function
                L  += mu *np.log10(np.e)*np.log(x)
                if not only_L:
                    G[index] += mu * np.log10(np.e)/x

    return [L,G]

#################################
## para_stg
#################################
def merge_stg(para_stgs):

    stg = {}
    stg['para_list']      = []
    stg['para_length']    = {}
    stg['para_exp']       = {}
    stg['para_ini']       = {}
    stg['para_step_Q']    = {}
    stg['para_step_diff'] = {}

    for para_stg in para_stgs:
        stg['para_list'].extend(para_stg['list'])
        stg['para_length'].update(para_stg['length'])
        stg['para_exp'].update(para_stg['exp'])
        stg['para_ini'].update(para_stg['ini'])
        stg['para_step_Q'].update(para_stg['step_Q'])
        stg['para_step_diff'].update(para_stg['step_diff'])

    return stg
