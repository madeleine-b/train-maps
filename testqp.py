from numpy.random import randn
import cvxpy as cvx
from qcqp import *

import numpy as np
import itertools

import time
import matplotlib.pyplot as plt

#np.random.seed(1)

n = 10

### write a matrix M here that is n-by-n and whose entries are 0 or 1. Also, M has to be symmetric, and dtype=int

#E = [np.sort(np.random.choice(range(i+1,n),np.random.choice(range(1,min(11,n-i)),replace=False))) for i in range(n-1)]
#M = np.zeros((n,n),dtype=int)
#for i in range(n-1):
#    for j in E[i]:
#        M[i,j] = 1
#M = M + M.T

#M = np.ones((n,n),dtype=int)

def ee(i,d): # i-th standard basis vector in R^d
    return np.array([int(k==i) for k in range(d)]).reshape(d,1)

def Delta(i,j,d): # 0-1 d-by-d matrix with 1 only at the (i,j)-th entry
    ans = np.zeros((d,d))
    ans[i,j] = 1
    return ans #ee(i,d) @ ee(j,d).T
    
def reverse_neighbor(i,j,neighbors): # finds k such that N_{i}(k) = j
    N_i = np.array(neighbors[i])
    return np.argwhere(N_i == j)[0,0]
    
def neighbor_list(M):
    n = M.shape[0]
    Mp = M.copy() # modified adjacency so that the Markov chain is ergodic
    d = np.sum(Mp,axis=1)
    #print('degrees:', d)
    #print('\n #nonzero weights:', np.sum(d))

    boundary = np.argwhere(d==1).flatten()
    for b in boundary: # make M irreducible
        Mp[b,b] = 1
    d = np.sum(Mp,axis=1)

    #w_supp = np.argwhere(Mp!=0)
    #n_w = len(w_supp)

    edges = [[i,j] for i,j in itertools.product(range(n),range(n)) if Mp[i,j] != 0] # the edges we care about (~ 300 out of ~ 140k)

    neighbors = [] # neighbor list: neighbors[i] are neighbors of vertex i
    q = 0
    for i in range(n):
        neighbors.append([edges[k][1] for k in range(q,q+d[i])])
        q += d[i]

    #print(neighbors)

    lengths = [0]
    q = 0
    for t in neighbors:
        q += len(t)
        lengths.append(q)
        
    R = [[reverse_neighbor(neighbors[j][k],j,neighbors) for k in range(len(neighbors[j]))] for j in range(n)]
        
    return neighbors,lengths,Mp,d,R

neighbors,lengths,M,d,R = neighbor_list(M)

lams = np.sort(np.linalg.eig(np.diag(d)-M)[0])[:3] # smallest 3 e-values of the Laplacian
print('lambdas:', lams)

m = lengths[-1] # number of edges
print('m:', m)

def constraint_matrices(m,n,neighbors,R,lengths):
    A1 = []
    A2 = []
    b1 = []
    b2 = []
    c3 = []
    c4 = np.array([0]*(2*m)+[1]*n)
    ms = [len(N) for N in neighbors]
    d = 2*m+n
    for i in range(n):
        A1.append(np.sum([Delta(lengths[i]+k,2*m+neighbors[i][k],d) for k in range(ms[i])], axis = 0))
        b1.append(-ee(2*m+i,d))
        
        a2_i = []
        b2_i = []
        for r in range(ms[i]):
            a2_i.append(np.sum([Delta(lengths[i]+r,m+lengths[neighbors[i][r]]+k,d) for k in range(ms[r])], axis = 0))
            b2_i.append(-ee(m+lengths[i]+r,d))
        A2.append(a2_i)
        b2.append(b2_i)
        
        c3.append(np.sum([ee(lengths[neighbors[i][k]]+R[i][k],d) for k in range(ms[i])], axis = 0).reshape(d))
    return A1,b1,A2,b2,c3,c4

A1,b1,A2,b2,c3,c4 = constraint_matrices(m,n,neighbors,R,lengths)
	
def symmetrize(A):
	return (A+A.T)/2
	

#pi = np.random.random(n)
#pi = pi/sum(pi)
pi = np.ones(n)/n

As = A1
for A in A2:
    As += A
bs = b1
for b in b2:
	bs += b
g = len(As)
print('no. cons.:',g)
C = np.vstack(c3 + [c4])

x = cvx.Variable(2*m+n)
obj = cvx.sum_squares(x[-n:]-pi)
consQ = [cvx.quad_form(x, As[i]) + bs[i].T @ x == 0 for i in range(g)] # Quadratic constrains
consL = [C @ x == np.ones((n+1,1))] # Linear constraints
cons = consQ + consL + [x >= np.zeros(2*m+n)]
prob = cvx.Problem(cvx.Minimize(obj), cons)

qcqp = QCQP(prob)
qcqp.suggest(SDR) # qcqp.suggest(SDR, solver=cvx.MOSEK)
print("SDR lower bound: %.3f" % qcqp.sdr_bound)
f_cd, v_cd = qcqp.improve(ADMM) # COORD_DESCENT
print("ADMM: objective %.3f, violation %.3f" % (f_cd, v_cd))

x1 = x.value
print(x1)
print('p:',x1[-n:])
print('pi:',pi)
print('error:',obj.value)
print([(x1.T @ As[i] @ x1 + bs[i].T @ x1)[0,0] for i in range(g)])
print(C @ x1 - np.ones((n+1,1)))