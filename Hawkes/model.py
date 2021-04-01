import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from scipy.special import gamma,digamma

from .tools import Quasi_Newton,merge_stg,loglinear_COS,plinear,pconst

try:
    import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()},language_level=2)
    from .Hawkes_C import LG_kernel_SUM_exp_cython, LG_kernel_SUM_pow_cython, preprocess_data_nonpara_cython
    cython_import = True
    #print("cython mode")
except:
    cython_import = False
    #print("failed to import cython: use python mode")


##########################################################################################################
## class
##########################################################################################################
class base_class:

    ### initialize
    def set_kernel(self,type,**kwargs):
        kernel_class = {'exp':kernel_exp, 'pow':kernel_pow, 'nonpara':kernel_nonpara}
        self.kernel = kernel_class[type](**kwargs)
        return self

    def set_baseline(self,type,**kwargs):
        baseline_class = {'const':baseline_const,'loglinear':baseline_loglinear,'plinear':baseline_plinear,'pconst':baseline_pconst,'custom':baseline_custom}
        self.baseline = baseline_class[type](**kwargs)
        return self

    def set_parameter(self,para):
        self.para = para
        self.baseline.set_parameter(para)
        self.kernel.set_parameter(para)
        return self

    def set_data(self,Data,itv):
        st,en = itv
        T = Data['T']
        T = T[ (st<T) & (T<en) ]
        Data = {'T':T}
        self.Data = Data
        self.itv = itv
        self.baseline.set_data(Data,itv)
        self.kernel.set_data(Data,itv)
        return self

    def set_itv(self,itv):
        self.itv = itv
        self.baseline.set_itv(itv)
        self.kernel.set_itv(itv)
        return self

    ### l
    def tl(self):
        if self.kernel.has_sequential:
            T = self.Data['T']
            itv = self.itv
            [t,l_kernel] = self.kernel.sequential().tl(T,itv)
            l_baseline = self.baseline.l(t)
            
        return [t,l_kernel+l_baseline,l_baseline]
    
    def t_trans(self):
        T = self.Data['T']
        itv = self.itv
        INT_iei = self.kernel.INT_iei() + self.baseline.INT_iei()
        T_ext_trans = INT_iei.cumsum()
        self.T_trans = T_ext_trans[:-1]
        self.itv_trans = [0,T_ext_trans[-1]]
        return [self.T_trans,self.itv_trans]

    ### branching ratio
    def branching_ratio(self):
        return self.kernel.branching_ratio()

    ### plot
    def plot_l(self):
        T = self.Data['T']
        [t,l,l_baseline] = self.tl()
        plot_l(T,t,l,l_baseline)

    def plot_N(self):
        T = self.Data['T']
        itv = self.itv
        plot_N(T,itv)
        
    def plot_KS(self):
        self.t_trans()
        plot_KS(self.T_trans,self.itv_trans)

class simulator(base_class):

    def simulate(self,itv):
        if not (self.kernel.type == 'exp' or self.kernel.type == 'pow'):
            sys.exit('A kernel function must be a exponential or power-law function.')
        self.set_itv(itv)
        l_kernel_sequential = self.kernel.sequential()
        l_baseline = self.baseline.l
        T = simulate(l_kernel_sequential,l_baseline,itv)
        self.Data = {'T':T}
        return T
    
def simulate(l_kernel_sequential,l_baseline,itv):

    N_MAX = 1000001
    T = np.empty(N_MAX,dtype='f8')
    [st,en] = itv
    x = st;
    l0 = l_baseline(st);
    i = 0;

    while 1:

        step = np.random.exponential()/l0
        x += step
        l_kernel_sequential.step_forward(step)
        l1 = l_baseline(x) + l_kernel_sequential.l

        if (x>en) or (i==N_MAX):
            break

        if np.random.rand() < l1/l0: ## Fire
            T[i] = x
            i += 1
            l_kernel_sequential.event()

        l0 = l_baseline(x) + l_kernel_sequential.l

    T = T[:i]

    return T

class estimator(base_class):

    def fit(self,T,itv,prior=[],opt=[],merge=[]):
        T = np.array(T); T = T[(itv[0]<T)&(T<itv[1])].copy();
        self.set_data({'T':T},itv)

        stg_b = self.baseline.prep_fit()
        stg_k = self.kernel.prep_fit()
        stg = merge_stg([stg_b,stg_k])
        self.stg = stg

        [para,L,ste,G_norm,i_loop] = Quasi_Newton(self,prior,merge,opt)

        self.para = para
        self.parameter = para
        self.L = L
        self.AIC = -2.0*(L-len(para))
        self.br = self.kernel.branching_ratio()
        self.ste = ste
        self.i_loop = i_loop

        return self

    def LG(self,para,only_L=False):

        self.set_parameter(para)

        [l_baseline,dl_baseline]     = self.baseline.LG_SUM()
        [Int_baseline,dInt_baseline] = self.baseline.LG_INT()
        [l_kernel,dl_kernel]         = self.kernel.LG_SUM()
        [Int_kernel,dInt_kernel]     = self.kernel.LG_INT()

        l = l_baseline + l_kernel
        Int = Int_baseline + Int_kernel
        dl = dict(list(dl_baseline.items())+list(dl_kernel.items()))
        dInt = dict(list(dInt_baseline.items())+list(dInt_kernel.items()))

        L = np.sum(np.log(l)) - Int
        G = { key: (dl[key]/l).sum(axis=-1) - dInt[key] for key in dl }

        return [L,G]

    def predict(self,en_f,num_seq=1):
        T = self.Data['T']
        itv = self.itv;
        l_kernel_sequential = self.kernel.sequential()
        l_baseline = self.baseline.l
        l_kernel_sequential.input_history([T,itv[1]])
        T_pred = []
        for i in range(num_seq):
            l_kernel_sequential.load_initial_state()
            T_pred.append( simulate(l_kernel_sequential,l_baseline,[itv[1],en_f]) )
        self.en_f = en_f
        self.T_pred = T_pred
        return T_pred

    def plot_N_pred(self):
        T = self.Data['T']
        T_pred = self.T_pred
        itv = self.itv
        en_f = self.en_f
        plot_N_pred(T,T_pred,itv,en_f)

##########################################################################################################
## base component class
##########################################################################################################
class base_component_class():

    def set_parameter(self,para):
        self.para = para
        return self

    def set_data(self,Data,itv):
        self.Data = Data
        self.itv = itv
        return self

    def set_itv(self,itv):
        self.itv = itv
        return itv
    
class base_component_baseline_class(base_component_class):
    
    def INT_iei(self):
        T = self.Data['T']
        st,en = self.itv
        mu = self.l(T)
        dT = np.ediff1d(np.hstack([st,T,en]))
        l_INT = np.hstack([mu[0],mu])*dT/2 + np.hstack([mu,mu[-1]])*dT/2
        return l_INT
    
class base_component_kernel_class(base_component_class):

    def LG_SUM(self):

        if self.has_sequential:
            [l,dl] = self.sequential(mode='estimation').LG_SUM(self.Data['T'])
        else:
            T = self.Data['T']
            n = len(T)
            l  = np.zeros(n)
            dl = { key: [] for key in self.para_list }

            for i in np.arange(1,n):
                l[i] = self.func(T[i]-T[:i]).sum()
                dl_i = self.d_func(T[i]-T[:i])
                for key in self.para_list:
                    dl[key].append( dl_i[key].sum(axis=-1) )
            
            dl = { key: [dl[key][0]*0] + dl[key] for key in dl }
            dl = { key: np.array(dl[key]).transpose() for key in dl }
            
        return [l,dl]

    def LG_INT(self):
        T = self.Data['T']
        [_,en] = self.itv
        Int  = self.int(0,en-T).sum()
        dInt_tmp = self.d_int(np.zeros_like(T),en-T)
        dInt = { key:dInt_tmp[key].sum(axis=-1) for key in self.para_list }
        return [Int,dInt]
    
    def INT_iei(self):
        
        T = self.Data['T']
        itv = self.itv
        st,en = itv
        n = len(T)
        T_ext = np.hstack([T,en])
        
        if self.has_sequential:
            l_INT = self.sequential().INT_iei(T,itv)
        else:
            l_INT = [ self.int(T_ext[i]-T_ext[:i+1],T_ext[i+1]-T_ext[:i+1]).sum() for i in range(n) ]
            l_INT = np.hstack([0,l_INT])
                                
        return l_INT
        
class kernel_sequential():

    def LG_SUM(self,T):

        n = T.shape[0]
        l  = np.zeros(n)
        dl = self.dl
        dl = { key:np.zeros(n) for key in dl }

        for i in range(n-1):
            self.event()
            self.step_forward(T[i+1]-T[i])
            l[i+1] = self.l
            dl_i = self.dl
            for key in dl_i:
                dl[key][i+1] = dl_i[key]

        return [l,dl]

    def tl(self,T,itv):

        m = 30
        n = len(T)
        [st,en] = itv

        t = np.hstack([ np.linspace(t[0],t[1],m) for t in np.vstack([np.hstack([st,T]),np.hstack([T,en])]).transpose() ])

        mark = np.zeros((n+1,m),dtype='i8')
        mark[:,m-1] = 1
        mark = mark.flatten()
        mark[-1] = 0

        l = np.zeros_like(t)

        for i in range(t.shape[0]-1):

            if mark[i] == 1:
                self.event()

            self.step_forward(t[i+1]-t[i])
            l[i+1] = self.l

        return [t,l]
    
    def INT_iei(self,T,itv):
        n = len(T)
        [st,en] = itv
        T_ext = np.hstack([st,T,en])
        l_INT = np.zeros(n+1)
        
        for i in range(n+1):
            self.step_forward(T_ext[i+1]-T_ext[i])
            l_INT[i] = self.Int
            self.event()
            
        return l_INT

    def input_history(self,history):
        T_ext = np.hstack(history) # [T,en] = history
        n = len(T_ext)-1
        self.g *= 0
        self.l *= 0
        for i in range(n):
            self.event()
            self.step_forward(T_ext[i+1]-T_ext[i])
        self.g0 = self.g
        self.l0 = self.l

    def load_initial_state(self):
        self.g = self.g0
        self.l = self.l0

##########################################################################################################
## baseline class
##########################################################################################################
class baseline_const(base_component_baseline_class):

    def __init__(self):
        self.type = 'const'

    def prep_fit(self):
        itv = self.itv
        T = self.Data['T']
        n = T.shape[0]

        list =      ['mu']
        length =    {'mu':1 }
        exp =       {'mu':True }
        ini =       {'mu':0.5*len(T)/(itv[1]-itv[0]) }
        step_Q =    {'mu':0.2 }
        step_diff = {'mu':0.01 }

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.para
        Data = self.Data
        n = len(Data['T'])
        mu = para['mu']
        l = mu*np.ones(n)
        dl = {'mu':np.ones(n)}
        return [l,dl]

    def LG_INT(self):
        para = self.para
        mu = para['mu']
        [st,en] = self.itv
        Int = mu*(en-st)
        dInt = {'mu':en-st}
        return [Int,dInt]

    def l(self,t):
        para = self.para
        mu = para['mu']
        return mu if isinstance(t,numbers.Number) else mu*np.ones_like(t)

###########################
class baseline_loglinear(base_component_baseline_class):

    def __init__(self,num_basis):
        self.type = 'loglinear'
        self.num_basis = num_basis

    def prep_fit(self):
        num_basis = self.num_basis;
        itv = self.itv
        T = self.Data['T']
        n = T.shape[0]

        list =      ['mu']
        length =    {'mu':num_basis }
        exp =       {'mu':False }
        ini =       {'mu':np.log( 0.5*n/(itv[1]-itv[0]) ) * np.ones(num_basis) }
        step_Q =    {'mu':0.2 * np.ones(num_basis) }
        step_diff = {'mu':0.01 * np.ones(num_basis) }

        self.loglinear = loglinear_COS(itv,num_basis).set_x(T)

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.para
        loglinear = self.loglinear
        coef = para['mu']
        loglinear.set_coef(coef)
        l = loglinear.get_y()
        dl = loglinear.get_dy()
        dl = {'mu':np.transpose(dl)}
        return [l,dl]

    def LG_INT(self):
        para = self.para
        loglinear = self.loglinear
        coef = para['mu']
        loglinear.set_coef(coef)
        Int = loglinear.get_int()
        dInt = loglinear.get_dint()
        dInt = {'mu':dInt}
        return [Int,dInt]

    def l(self,t):
        para = self.para
        loglinear = self.loglinear
        coef = para['mu']
        return loglinear.set_coef(coef).get_y_at(t)

###########################
class baseline_plinear(base_component_baseline_class):

    def __init__(self,num_basis):
        self.type = 'plinear'
        self.num_basis = num_basis

    def prep_fit(self):
        itv = self.itv
        T = self.Data['T']
        n = T.shape[0]
        num_basis = self.num_basis

        list   =    ['mu']
        length =    {'mu':num_basis }
        exp =       {'mu':True }
        ini =       {'mu':0.5*n/(itv[1]-itv[0])*np.ones(num_basis) }
        step_Q =    {'mu':0.2 * np.ones(num_basis) }
        step_diff = {'mu':0.01 * np.ones(num_basis) }

        self.plinear = plinear(itv,num_basis).set_x(T)

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.para
        plinear = self.plinear
        coef = para['mu']
        plinear.set_coef(coef)
        l = plinear.get_y()
        dl = plinear.get_dy()
        dl = {'mu':np.transpose(dl)}
        return [l,dl]

    def LG_INT(self):
        para = self.para
        plinear = self.plinear
        coef = para['mu']
        plinear.set_coef(coef)
        Int = plinear.get_int()
        dInt = plinear.get_dint()
        dInt = {'mu':dInt}
        return [Int,dInt]

    def l(self,t):
        para = self.para
        plinear = self.plinear
        coef = para['mu']
        return plinear.set_coef(coef).get_y_at(t)
    
###########################
class baseline_pconst(base_component_baseline_class):

    def __init__(self,num_basis):
        self.type = 'pconst'
        self.num_basis = num_basis

    def prep_fit(self):
        itv = self.itv
        T = self.Data['T']
        n = T.shape[0]
        num_basis = self.num_basis

        list   =    ['mu']
        length =    {'mu':num_basis }
        exp =       {'mu':True }
        ini =       {'mu':0.5*n/(itv[1]-itv[0])*np.ones(num_basis) }
        step_Q =    {'mu':0.2 * np.ones(num_basis) }
        step_diff = {'mu':0.01 * np.ones(num_basis) }

        self.pconst = pconst(itv,num_basis).set_x(T)

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.para
        pconst = self.pconst
        coef = para['mu']
        pconst.set_coef(coef)
        l = pconst.get_y()
        dl = pconst.get_dy()
        dl = {'mu':np.transpose(dl)}
        return [l,dl]

    def LG_INT(self):
        para = self.para
        pconst = self.pconst
        coef = para['mu']
        pconst.set_coef(coef)
        Int = pconst.get_int()
        dInt = pconst.get_dint()
        dInt = {'mu':dInt}
        return [Int,dInt]

    def l(self,t):
        para = self.para
        pconst = self.pconst
        coef = para['mu']
        return pconst.set_coef(coef).get_y_at(t)

###########################
class baseline_custom(base_component_baseline_class):

    def __init__(self,l_custom):
        self.type = 'custom'
        self.l_custom = l_custom

    def prep_fit(self):
        list   =    []
        length =    {}
        exp =       {}
        ini =       {}
        step_Q =    {}
        step_diff = {}

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}
    
    def LG_SUM(self):
        l = self.l(self.Data['T'])
        dl = {}
        return [l,dl]

    def LG_INT(self):
        Int = self.INT_iei().sum()
        dInt = {}
        return [Int,dInt]
    
    def l(self,t):
        return self.l_custom(t)
    

##########################################################################################################
## kernel class
##########################################################################################################
class kernel_exp(base_component_kernel_class):

    def __init__(self,num_exp=1):
        self.type = 'exp'
        self.num_exp = num_exp
        self.para_list = list( itertools.product(['alpha','beta'],range(num_exp)))
        self.has_sequential = True

    def prep_fit(self):
        num_exp = self.num_exp
        list =      ['alpha','beta']
        length =    {'alpha':num_exp,                                                     'beta':num_exp               }
        exp =       {'alpha':True,                                                        'beta':True                  }
        ini =       {'alpha':(np.arange(num_exp)+1.0)*0.5/np.sum(np.arange(num_exp)+1.0), 'beta':np.ones(num_exp)      }
        step_Q =    {'alpha':np.ones(num_exp)*0.2,                                        'beta':np.ones(num_exp)*0.2  }
        step_diff = {'alpha':np.ones(num_exp)*0.01,                                       'beta':np.ones(num_exp)*0.01 }
        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        
        if cython_import:
            para = self.para
            alpha = np.array(para['alpha']).flatten()
            beta  = np.array(para['beta']).flatten()
            [l,dl] = LG_kernel_SUM_exp_cython(self.Data['T'], alpha, beta)
        else:
            [l,dl] = super().LG_SUM()

        return [l,dl]

    def func(self,x):
        para = self.para
        num_exp = self.num_exp
        alpha = np.array(para['alpha']).flatten()
        beta  = np.array(para['beta']).flatten()

        l = 0
        for i in range(num_exp):
            l = l + alpha[i] * beta[i] * np.exp( -beta[i] * x )

        return l

    def d_func(self,x):
        para = self.para
        num_exp = self.num_exp
        alpha = np.array(para['alpha']).flatten()
        beta  = np.array(para['beta']).flatten()

        dl = {}
        for i in range(num_exp):
            dl[('alpha',i)] = np.exp( -beta[i] * x ) * beta[i]
            dl[('beta',i) ] = np.exp( -beta[i] * x ) * ( alpha[i] - alpha[i] * beta[i] * x )

        return dl

    def int(self,x1,x2):
        para = self.para
        num_exp = self.num_exp
        alpha = np.array(para['alpha']).flatten()
        beta  = np.array(para['beta']).flatten()

        Int = 0
        for i in range(num_exp):
            Int = Int + alpha[i] * ( np.exp( -beta[i] * x1 ) - np.exp( -beta[i] * x2 ) )

        return Int

    def d_int(self,x1,x2):
        para = self.para
        num_exp = self.num_exp
        alpha = np.array(para['alpha']).flatten()
        beta  = np.array(para['beta']).flatten()

        dInt = {}
        for i in range(num_exp):
            dInt[('alpha',i)] = np.exp( -beta[i] * x1 ) - np.exp( -beta[i] * x2 )
            dInt[('beta',i) ] = alpha[i] * ( - x1 * np.exp( -beta[i] * x1 ) + x2 * np.exp( -beta[i] * x2 ) )

        return dInt

    def branching_ratio(self):
        para = self.para
        br = np.array(para['alpha']).sum()
        return br

    def sequential(self,mode='simulation'):
        para = self.para
        num_exp = self.num_exp
        return kernel_sequential_exp(para,num_exp,mode=mode)

class kernel_sequential_exp(kernel_sequential):
    
    def __init__(self,para,num_exp,mode='simulation'):
        self.mode = mode
        self.alpha = np.array(para['alpha']).flatten()
        self.beta  = np.array(para['beta']).flatten()
        self.num_exp = num_exp
        self.g   = np.zeros(num_exp)
        self.l = 0
        self.Int = 0
        
        if self.mode == 'estimation':
            self.g_b = np.zeros(num_exp)
            self.dl = { key:0 for key in itertools.product(['alpha','beta'],range(num_exp)) }
       
    def step_forward(self,step):
        alpha = self.alpha; beta = self.beta; num_exp = self.num_exp;
        g = self.g
        r = np.exp(-beta*step)
        Int = ( g*(1-r)/beta ).sum()
        g = g*r
        l = g.sum()
        self.g = g
        self.l = l
        self.Int = Int
        
        if self.mode == 'estimation':
            g_b = self.g_b
            g_b = g_b*r - g*step
            dl = { ('alpha',i):g[i]/alpha[i] for i in range(num_exp) }
            dl.update( { ('beta',i): g_b[i]  for i in range(num_exp) } )
            self.g_b = g_b
            self.dl = dl

        return self

    def event(self):
        alpha = self.alpha; beta = self.beta;
        g = self.g
        g = g + alpha*beta
        l = g.sum()
        self.g  = g
        self.l = l
        
        if self.mode == 'estimation':
            g_b = self.g_b
            g_b = g_b + alpha
            self.g_b = g_b

        return self

###########################
class kernel_pow(base_component_kernel_class):

    def __init__(self):
        self.type = 'pow'
        self.para_list = ['k','p','c']
        self.has_sequential = True

    def prep_fit(self):
        list = ['k','p','c']
        length =    {'k': 1,    'p':1,    'c':1    }
        exp =       {'k': True, 'p':True, 'c':True }
        ini =       {'k': 0.25, 'p':1.5,  'c':1.0  }
        step_Q =    {'k': 0.2,  'p':0.2,  'c':0.2  }
        step_diff = {'k': 0.01, 'p':0.01, 'c':0.01 }
        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def func(self,x):
        para = self.para
        k = para['k']; p = para['p']; c = para['c'];
        return k * (x+c)**(-p)

    def d_func(self,x):
        para = self.para
        k = para['k']; p = para['p']; c = para['c'];
        dl = {}
        dl['k']     =      (x+c)**(-p)
        dl['p']     = -k * (x+c)**(-p) * np.log(x+c)
        dl['c']     = -k * (x+c)**(-p-1) * p
        return dl

    def int(self,x1,x2):
        para = self.para
        k = para['k']; p = para['p']; c = para['c'];
        Int = k / (-p+1) * ( (x2+c)**(-p+1) - (x1+c)**(-p+1) )
        return Int

    def d_int(self,x1,x2):
        para = self.para
        k = para['k']; p = para['p']; c = para['c'];
        f1 = k / (-p+1) * (x1+c)**(-p+1)
        f2 = k / (-p+1) * (x2+c)**(-p+1)
        dInt = {}
        dInt['k']     = (f2-f1)/k
        dInt['p']     = (f2-f1)/(-p+1) + ( - f2*np.log(x2+c) + f1*np.log(x1+c) )
        dInt['c']     = ( f2/(x2+c) - f1/(x1+c) ) * (-p+1)
        return dInt

    def branching_ratio(self):
        para = self.para
        k = para['k']; p = para['p']; c = para['c'];
        br = -k/(-p+1)*c**(-p+1) if p>1 else np.inf
        return br

    def sequential(self,mode='simulation'):
        para = self.para
        return kernel_sequential_pow(para,mode=mode)

class kernel_sequential_pow(kernel_sequential):

    def __init__(self,para,mode='simulation'):
        self.mode = mode
        k = para['k']; p = para['p']; c = para['c'];
        num_div = 16
        delta = 1.0/num_div
        s = np.linspace(-9,9,num_div*18+1)
        log_phi = s-np.exp(-s)
        log_dphi = log_phi + np.log(1+np.exp(-s))
        phi = np.exp(log_phi)   # phi = np.exp(s-np.exp(-s))
        H   = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
        g = np.zeros_like(s)
        self.g = g
        self.l = 0
        self.Int = 0
        self.phi = phi
        self.H = H
        
        if self.mode == 'estimation':
            H_k = delta *     np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
            H_p = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p) * (log_phi-digamma(p))
            H_c = delta * k * np.exp( log_dphi +     p*log_phi - c*phi ) / gamma(p) * (-1)
            self.H_k = H_k
            self.H_p = H_p
            self.H_c = H_c
            self.dl = {'k':0, 'p':0, 'c':0}
        
    def step_forward(self,step):
        g = self.g; phi = self.phi; H = self.H; 
        index = phi*step<1e-6
        v = np.where(index,step,np.divide((1-np.exp(-phi*step)),phi,where=~index))
        Int = (g*v).dot(H) #(g*(1-np.exp(-phi*step))/phi).dot(H)
        g = g*np.exp(-phi*step)
        l = g.dot(H)
        self.g = g
        self.l = l
        self.Int = Int
        
        if self.mode == 'estimation':
            H_k = self.H_k; H_p = self.H_p; H_c = self.H_c;
            dl = {'k':g.dot(H_k),'p':g.dot(H_p),'c':g.dot(H_c)}
            self.dl = dl

        return self

    def event(self):
        g = self.g; H = self.H;
        g = g+1.0
        l = g.dot(H)
        self.g = g
        self.l = l

        return self
    
###############################
class kernel_nonpara(base_component_kernel_class):

    def __init__(self,support,num_bin):
        self.type = 'nonpara'
        self.support = support
        self.num_bin = num_bin
        self.bin_edge = np.linspace(0,support,num_bin+1)
        self.bin_width = support/num_bin
        self.para_list = ['g']
        self.has_sequential = False
        
    def set_data(self,Data,itv):
        self.Data = Data
        self.itv = itv
        dl,dInt = preprocess_data_nonpara_cython(Data['T'],self.bin_edge,itv[1])
        self.dl = dl
        self.dInt = dInt
        return self
        
    def prep_fit(self):
        num_bin = self.num_bin
        support = self.support
        list =      ['g']
        length =    {'g': num_bin }
        exp =       {'g': True }
        ini =       {'g': np.ones(num_bin)*0.5/support}
        step_Q =    {'g': np.ones(num_bin)*0.2} 
        step_diff = {'g': np.ones(num_bin)*0.01}
        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}
    
    def LG_SUM(self):
        g = self.para['g']
        l = np.dot(g,self.dl)
        dl = {'g':self.dl}
        return [l,dl]
    
    def LG_INT(self):
        g = self.para['g']
        Int = g.dot(self.dInt)
        dInt = {'g':self.dInt}
        return [Int,dInt]
        
    def func(self,x):
        bin_edge = self.bin_edge
        g = self.para['g']
        g_ext = np.hstack([g,0])
        l = g_ext[ np.searchsorted(bin_edge,x,side='right') - 1 ]
        return l

    def d_func(self,x):
        bin_edge = self.bin_edge
        dl = np.zeros((bin_edge.shape[0],x.shape[0]))
        dl[np.searchsorted(bin_edge,x,side='right')-1,np.arange(x.shape[0])] = 1.0
        dl = {'g': dl[:-1] }
        return dl
    
    def int(self,x1,x2):
        bin_edge = self.bin_edge
        bin_width = self.bin_width
        g = self.para['g']
        g_ext = np.hstack([g,0])
        cum = np.hstack([0,g.cumsum()*bin_width])
        index1 = np.searchsorted(bin_edge,x1,side='right') - 1
        index2 = np.searchsorted(bin_edge,x2,side='right') - 1
        int1 = cum[index1] + (x1-bin_edge[index1]) * g_ext[index1]
        int2 = cum[index2] + (x2-bin_edge[index2]) * g_ext[index2]
        return int2-int1
    
    def d_int(self,x1,x2):
        bin_edge = self.bin_edge
        bin_width = self.bin_width
        index1 = np.searchsorted(bin_edge,x1,side='right') - 1
        index2 = np.searchsorted(bin_edge,x2,side='right') - 1
        dInt1 = np.vstack([ np.hstack([np.ones(index)*bin_width,x1[i]-bin_edge[index],np.zeros(bin_edge.shape[0]-index-1)]) for i,index in enumerate(index1) ]).transpose()
        dInt2 = np.vstack([ np.hstack([np.ones(index)*bin_width,x2[i]-bin_edge[index],np.zeros(bin_edge.shape[0]-index-1)]) for i,index in enumerate(index2) ]).transpose()
        
        return {'g': (dInt2 - dInt1)[:-1] }
        
    def branching_ratio(self):
        br = self.para['g'].sum()*self.bin_width
        return br
    
    def plot(self):
        bin_edge = self.bin_edge
        x = np.vstack([bin_edge[:-1],bin_edge[1:]]).transpose().flatten()
        y = np.repeat(self.para['g'],2)
        plt.plot(x,y,'k-')

###########################################################################################
###########################################################################################
## graph routine
###########################################################################################
###########################################################################################
def plot_N(T,itv):

    gs = gridspec.GridSpec(100,1)

    plt.figure(figsize=(4,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    [st,en] = itv
    n = len(T)
    x = np.hstack([st,np.repeat(T,2),en])
    y = np.repeat(np.arange(n+1),2)

    plt.subplot(gs[0:10,0])
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.array( [0,1,np.NaN] * n ),'k-',linewidth=0.5)
    plt.xticks([])
    plt.xlim(itv)
    plt.ylim([0,1])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.subplot(gs[15:100,0])
    plt.plot(x,y,'k-',clip_on=False)
    plt.xlim(itv)
    plt.ylim([0,n])
    plt.xlabel('time')
    plt.ylabel(r'$N(0,t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_l(T,x,l,l_baseline):

    gs = gridspec.GridSpec(100,1)

    plt.figure(figsize=(4,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    l_max = l.max()
    n = len(T)

    plt.subplot(gs[0:10,0])
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.array( [0,1,np.NaN] * n ),'k-',linewidth=0.5)
    plt.xticks([])
    plt.xlim([x[0],x[-1]])
    plt.ylim([0,1])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.subplot(gs[15:100,0])
    plt.plot(x,l,'k-',lw=1)
    plt.plot(x,l_baseline,'k:',lw=1)
    plt.xlim([x[0],x[-1]])
    plt.ylim([0,l_max])
    plt.xlabel('time')
    plt.ylabel(r'$\lambda(t|H_t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_N_pred(T,T_pred,itv,en_f):

    gs = gridspec.GridSpec(100,1)

    plt.figure(figsize=(4,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    [st,en] = itv
    n = len(T)
    x = np.hstack([st,np.repeat(T,2),en])
    y = np.repeat(np.arange(n+1),2)
    n_pred_max = np.max([ len(T_i) for T_i in T_pred ])

    plt.subplot(gs[0:10,0])
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.array( [0,1,np.NaN] * n ),'k-',linewidth=0.5)
    plt.xticks([])
    plt.xlim([itv[0],en_f])
    plt.ylim([0,1])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.subplot(gs[15:,0])
    plt.plot(x,y,'k-')
    plt.plot([en,en],[0,n+n_pred_max],'k--')

    for i in range(len(T_pred)):
        n_pred = len(T_pred[i])
        x = np.hstack([en,np.repeat(T_pred[i],2),en_f])
        y = np.repeat(np.arange(n_pred+1),2) + n
        plt.plot(x,y,'-',color=[0.7,0.7,1.0],lw=0.5)

    plt.xlim([st,en_f])
    plt.ylim([0,n+n_pred_max])
    plt.xlabel('time')
    plt.ylabel(r'$N(0,t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_KS(T_trans,itv_trans):
    from scipy.stats import kstest

    plt.figure(figsize=(4,4), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    n = len(T_trans)
    [st,en] = itv_trans
    x = np.hstack([st,np.repeat(T_trans,2),en])
    y = np.repeat(np.arange(n+1),2)/n
    w = 1.36/np.sqrt(n)
    [_,pvalue] = kstest(T_trans/itv_trans[1],'uniform')

    plt.plot(x,y,"k-",label='Data')
    plt.fill_between([0,n*w,n*(1-w),n],[0,0,1-2*w,1-w],[w,2*w,1,1],color="#dddddd",label='95% interval')
    plt.xlim([0,n])
    plt.ylim([0,1])
    plt.ylabel("cumulative distribution function")
    plt.xlabel("transfunced time")
    plt.title("p-value = %.3f" % pvalue)
    plt.legend(loc="upper left")
