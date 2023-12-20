import numpy as np
from scipy.linalg import solve_triangular

def cg(A, b, tol=1e-12):
    m, n = A.shape
    x = np.zeros(n)
    r = b - np.dot(A, x)
    p = r
    r_b = [np.linalg.norm(r) / np.linalg.norm(b)]
    for k in range(m):
        Ap = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        beta = np.dot(r, r) / np.dot(r_b[-1], r_b[-1])
        p = r + beta * p
        r_b.append(np.linalg.norm(r) / np.linalg.norm(b))
        if r_b[-1] < tol:
            break
    return x, r_b



# def arnoldi_n(A, Q, P):
#     # n-th step of arnoldi
#     m, n = Q.shape
#     q = np.zeros(m, dtype=Q.dtype)
#     h = np.zeros(n + 1, dtype=A.dtype)
#     q1 = b / np.linalg.norm(b)
#     for n in range(m):
#         v = Aq[n]
#         for j in range(n):
#             h[j][n] = np.conjugate(np.transpose(q), v)
#             h[n + 1][n] = np.linalg.norm(v)
#             q[n + 1] = v / h[n + 1][n]
#
#
#     return h, q


def arnoldi_n(A, Q, P):
    m, n = Q.shape
    q = np.dot(A, Q[:, n-1])
    for i in range(n):
        h = np.dot(Q[:, i].T, q)
        q -= h * Q[:, i]
    h = np.zeros((n+1, n))
    h[0:n, n-1] = np.dot(Q.T, q)
    q /= np.linalg.norm(q)
    return h, q

def gmres(A, b, P, tol):
    m, n = A.shape
    x = np.zeros(n)
    r = b - np.dot(A, x)
    r_b = [np.linalg.norm(r) / np.linalg.norm(b)]
    Q = np.zeros((m, n))
    Q[:, 0] = r / r_b[-1]
    for k in range(n):
        H, q = arnoldi_n(A, Q, P)
        Q = np.hstack((Q, q.reshape((m, 1))))
        Q, R = qr(Q)
        y = solve_triangular(R[0:k+2, 0:k+2], np.dot(Q.T, b))
        x = x + np.dot(Q[:, 0:k+2], y)
        r = b - np.dot(A, x)
        r_b.append(np.linalg.norm(r) / np.linalg.norm(b))
        if r_b[-1] < tol:
            break
    return x, r_b


# def gmres(A, b, P=None, tol=1e-5):
#     m = A.shape[0]
#     x = np.zeros(b.shape)
#     r_b = [np.linalg.norm(b - np.dot(A, x)) / np.linalg.norm(b)]
#     Q = np.zeros((m, m))
#     for i in range(m):

#
# def gmres(A, b, P=np.eye(0), tol=1e-12):
#     m = A.shape[0]
#     if P.shape != A.shape:
#         # default preconditioner P = I
#         P = np.eye(m)
#     x = np.zeros(m, dtype=b.dtype)
#     r_b = [1]
#     # todo
#
#     return x, r_b