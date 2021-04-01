import numpy as np
from scipy.special import gamma,digamma
cimport numpy as np

def LG_kernel_SUM_exp_cython(np.ndarray[np.float64_t,ndim=1] T, np.ndarray[np.float64_t,ndim=1] alpha, np.ndarray[np.float64_t,ndim=1] beta):
    cdef int m = len(alpha)
    cdef int n = T.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] l    = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] l_i  = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_a = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_b = np.zeros(n, dtype=np.float64)
    cdef dict dl = {}
    
    for i in range(m):
        l_i,dl_a,dl_b = LG_kernel_SUM_exp_i_cython(T,alpha[i],beta[i])
        l = l + l_i
        dl.update({('alpha',i):dl_a,('beta',i):dl_b})
        
    return [l,dl]
        

def LG_kernel_SUM_exp_i_cython(np.ndarray[np.float64_t,ndim=1] T, double alpha, double beta):

    cdef int n = T.shape[0]

    cdef np.ndarray[np.float64_t,ndim=1] l    = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_a = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_b = np.zeros(n, dtype=np.float64)

    cdef np.ndarray[np.float64_t,ndim=1] dt = T[1:] - T[:-1]
    cdef np.ndarray[np.float64_t,ndim=1] r = np.exp(-beta*dt)

    cdef double x = 0.0
    cdef double x_a = 0.0
    cdef double x_b = 0.0

    cdef int i;

    for i in range(n-1):
        x   = ( x   + alpha*beta  ) * r[i]
        x_a = ( x_a +       beta  ) * r[i]
        x_b = ( x_b + alpha       ) * r[i] - x*dt[i]

        l[i+1] = x
        dl_a[i+1] = x_a
        dl_b[i+1] = x_b

    return [l,dl_a,dl_b]


def LG_kernel_SUM_pow_cython(np.ndarray[np.float64_t,ndim=1] T,double k, double p, double c):

    cdef int n = T.shape[0]

    cdef np.ndarray[np.float64_t,ndim=1] l    = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_p = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_k = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_c = np.zeros(n, dtype=np.float64)

    cdef int num_div = 16
    cdef double delta = 1.0/num_div
    cdef np.ndarray[np.float64_t,ndim=1] s = np.linspace(-9,9,num_div*18+1)
    cdef np.ndarray[np.float64_t,ndim=1] log_phi = s-np.exp(-s)
    cdef np.ndarray[np.float64_t,ndim=1] log_dphi = log_phi + np.log(1+np.exp(-s))
    cdef np.ndarray[np.float64_t,ndim=1] phi = np.exp(log_phi)   # phi = np.exp(s-np.exp(-s))
    cdef np.ndarray[np.float64_t,ndim=1] dphi = np.exp(log_dphi) # dphi = phi*(1+np.exp(-s))

    cdef np.ndarray[np.float64_t,ndim=1] H      = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
    cdef np.ndarray[np.float64_t,ndim=1] H_p = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p) * (log_phi-digamma(p))
    cdef np.ndarray[np.float64_t,ndim=1] H_c = delta * k * np.exp( log_dphi +     p*log_phi - c*phi ) / gamma(p) * (-1)

    cdef np.ndarray[np.float64_t,ndim=1] g = np.zeros_like(s)

    cdef int i

    for i in range(n-1):
        g = (g+1)*np.exp( - phi*(T[i+1]-T[i]) )
        l[i+1] = g.dot(H)
        dl_k[i+1] = l[i+1]/k
        dl_p[i+1] = g.dot(H_p)
        dl_c[i+1] = g.dot(H_c)

    return [l,dl_k,dl_p,dl_c]

def preprocess_data_nonpara_cython(np.ndarray[np.float64_t,ndim=1] T, np.ndarray[np.float64_t,ndim=1] bin_edge, double en):
    
    cdef double support = bin_edge[-1]
    cdef double bin_width = bin_edge[1] - bin_edge[0]
    cdef int i,j
    cdef int n = T.shape[0]
    cdef int m = bin_edge.shape[0] - 1 # the number of bins
    
    ###### dl
    cdef list index_tgt_list = []
    cdef list index_trg_list = []
    
    for i in range(n):
        for j in range(i-1,-1,-1):
            if T[i] - T[j] < support:
                index_tgt_list.append(i)
                index_trg_list.append(j)
            else:
                break
                
    cdef np.ndarray[np.int64_t,ndim=1] index_tgt = np.array(index_tgt_list)
    cdef np.ndarray[np.int64_t,ndim=1] index_trg = np.array(index_trg_list)
    cdef np.ndarray[np.int64_t,ndim=1] index_bin = np.searchsorted(bin_edge,T[index_tgt]-T[index_trg],side='right') - 1 
    
    ###
    cdef np.ndarray[np.float64_t,ndim=2] dl = np.zeros((m,n))
    
    for i in range(index_tgt.shape[0]):
        dl[index_bin[i],index_tgt[i]] += 1.0 
        
    ###### dInt
    cdef np.ndarray[np.float64_t,ndim=2] dInt = np.zeros((m,n))
    cdef np.ndarray[np.int64_t,ndim=1] index = np.searchsorted(bin_edge,en-T,side='right') - 1
    cdef np.ndarray[np.float64_t,ndim=1] d_from_left = en - T - bin_edge[index]
    cdef int index_i
    
    for i in range(n):
        index_i = index[i]
        if index_i < m:
            dInt[index_i,i] = d_from_left[i] 
        if index_i > 0:
            for j in range(index_i):
                dInt[j,i] = bin_width
                
    return [dl,dInt.sum(axis=-1)]
