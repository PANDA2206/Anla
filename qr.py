import numpy as np


def sn(p):
        if (p == 0):
            s = 1
            return(s)
        else:
            s = p/np.abs(p)
            return(s)

def implicit_qr(A):
    m,n = A.shape
    W = np.zeros((m,n))
    for k in range(n):
        v = A[k:,k].copy()
        v[0] = v[0] + sn(v[0]) * np.linalg.norm(v)
        v = v / np.linalg.norm(v)


        #e1 =np.zeros(x.shape)
        #e1[0] = 1
    #3normx = np.linalg.norm(x)
       # v = ((sn(x[0])(normx)(e1))+ x)
       # v = v/(np.linalg.norm(v))

        W[k:, k] = v
        A[k:m,k:n] = np.matmul(I,A[k:m,k:n])
    R = A.astype(A)
    print(W)
    return (W, R)
#print(implicit_qr(A))
#W = implicit_qr(A)[0]

def form_q(W):
    m,n = W.shape
    I = np.identity(m)
    for col in range(1,m+1):
        for k in range(1,n+1):
             I[k-1:m,col-1] = ((I[k-1:m,col-1])- ((np.outer(W[k-1:m,k-1],W[k-1:m,k-1]))*2)@((I[k-1:m,col-1])))
    Q = np.transpose(I)
    return(Q)
#[W,R] = implicit_qr(A)
#Q= form_q(W)

#print(Q@R)
   #print(form_q(implicit_qr(A)[0]))