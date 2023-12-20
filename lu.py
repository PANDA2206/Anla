import numpy as np
#A = np.array([[2,7,1],[3,-2,0],[1,5,3]])
def lu(A):
    u = A.astype(float)
    m = A.shape[0]

    l = np.eye(m,m)
    for k in range(m-1):
        for j in range(k+1,m):
            l[j][k] = u[j][k] / u[k][k]
            u[j,k:m] = u[j,k:m] - l[j][k]*u[k,k:m]

    return l,u
#print(lu(A))
def maxabs_idx(A):
    (i,j) =(0,0)
    max = np.absolute(A[0][0])
    m,n = A.shape
    for k in range(m):
        for l in range(n):
            if np.absolute(A[k][l]) >max:
                (i, j) = (k, l)
                max = np.absolute(A[k][l])


    #max_abs_value = lst[0]
   # for num in lst:
       # if abs(num) > max_abs_value:
            #max_abs_value = abs(num)

    return (i, j)
#print(maxabs_idx(A))

def lu_complete(A):
    U = A.astype(float)
    m = A.shape[0]
    L = np.eye(m)
    P = np.eye(m)
    Q = np.eye(m)
    '''for k in range(m - 1):
         for l in range(k+1,m):
            l[j][k] = u[j][k] / u[k][k]
            u[j, k:m] = u[j, k:m] - l[j][k] * u[k, k:m]'''

    for k in range(m - 1):
        i = (maxabs_idx(U[k:m, k:m]))[0] + k
        j = (maxabs_idx(U[k:m, k:m]))[1] + k
        #    print(i,j,(maxabs_idx(U[k:m,k:m]))[2])

        # Swapping rows of U
        u = U[k, k:m].copy()
        U[k, k:m] = U[i, k:m]
        U[i, k:m] = u

        # Swapping corresponding rows in L
        u = L[k, 0:k].copy()  # columns 0:k instead of 1:k-1 as matrix first index is 0 in python and k is already k-1
        L[k, 0:k] = L[i, 0:k]
        L[i, 0:k] = u

        # Swapping columns of U
        u = U[:, k].copy()
        U[:, k] = U[:, j]
        U[:, j] = u


        # Swapping corresponding cols in L
        u = L[0:k, k].copy()  # rowa k:0 instead of k-1:1 as matrix first index is 0 in python and k is already k-1
        L[0:k, k] = L[0:k, j]
        L[0:k, j] = u

        u = P[k, :].copy()
        P[k, :] = P[i, :]
        P[i, :] = u

        u = Q[:, k].copy()
        Q[:, k] = Q[:, j]
        Q[:, j] = u


        for l in range(k + 1, m):
            # if U[l][k]>0:
            L[l][k] = U[l][k] / U[k][k]
            U[l, k:m] = U[l, k:m] - L[l][k] * (U[k, k:m])
        #print(L)
        #print(U)


    return P,Q,L,U

    #U = None
    #L = None
    #P = None
    #Q = None

