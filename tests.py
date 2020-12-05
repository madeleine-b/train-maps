from numpy.random import randn
import cvxpy as cvx
from qcqp import *

import numpy as np
import itertools

#n, m = 10, 15
#A = randn(m, n)
#b = randn(m, 1)

## Form a nonconvex problem.
#x = cvx.Variable(n)
#obj = cvx.sum_squares(A @ x - b)
#cons = [cvx.square(x) == 1]
#prob = cvx.Problem(cvx.Minimize(obj), cons)

## Create a QCQP handler.
#qcqp = QCQP(prob)

## Solve the SDP relaxation and get a starting point to a local method
#qcqp.suggest(SDR)
#print("SDR lower bound: %.3f" % qcqp.sdr_bound)

## Attempt to improve the starting point given by the suggest method
#f_cd, v_cd = qcqp.improve(COORD_DESCENT)
#print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
#print(x.value)


def ee(i,d):
    return np.array([int(k==i) for k in range(d)]).reshape(d,1)

def Delta(i,d):
    return ee(i,d) @ ee(d-1,d).T

def AA(i,n):
    return np.kron(Delta(i,2*n+1),np.eye(n,dtype=int))

def bb(i,n):
    return -ee(2*n**2+i,2*n**2+n)
    
#def Aqh(n):
#	ans = np.zeros((2*n**2+n,2*n**2+n))
#	ans[:2*n**2,:2*n**2] = np.kron(np.array([[0,1],[0,0]]),np.kron(np.ones((n,1)),np.kron(np.eye(n),np.ones((1,n)))))
#	return ans

def Aqh(i,j,n):
	ans = np.zeros((2*n**2+n,2*n**2+n))
	ans[i*n+j,n*(n+j):n*(n+j+1)] = np.ones(n)
	return ans
	
def bqh(i,j,n):
	return -ee(n*(n+i)+j,2*n**2+n)
	
def cq(j,n):
	return np.kron(np.hstack([np.ones(n),np.zeros(n+1)]),ee(j,n).reshape(n))
	
def symmetrize(A):
	return (A+A.T)/2
    
    
n = 7
#np.random.seed(1)
pi = np.random.random(n)
pi = pi/sum(pi)
As = [symmetrize(AA(i,n)) for i in range(n)]
Aqhs = [symmetrize(Aqh(i,j,n)) for i,j in itertools.product(range(n),range(n))]
b1s = [bb(i,n) for i in range(n)]
b2s = [bqh(i,j,n) for i,j in itertools.product(range(n),range(n))]
C1 = np.vstack([cq(j,n) for j in range(n)] + [np.hstack([np.zeros(2*n**2),np.ones(n)])])

x = cvx.Variable(2*n**2+n)
obj = cvx.sum_squares(x[-n:]-pi)
cons1 = [cvx.quad_form(x, As[i]) + b1s[i].T @ x == 0 for i in range(n)]
cons2 = [cvx.quad_form(x, Aqhs[i]) + b2s[i].T @ x == 0 for i in range(n)]
cons3 = [C1 @ x == np.ones((n+1,1))]
cons = cons1 + cons2 + cons3 + [x >= np.zeros(2*n**2+n)]
prob = cvx.Problem(cvx.Minimize(obj), cons)

## prob.solve()
qcqp = QCQP(prob)
qcqp.suggest(SDR) # qcqp.suggest(SDR, solver=cvx.MOSEK)
print("SDR lower bound: %.3f" % qcqp.sdr_bound)
f_cd, v_cd = qcqp.improve(ADMM) # COORD_DESCENT
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
print(x.value)
print(pi)
x1 = x.value
print(obj.value)
print([x1.T @ As[i] @ x1 + b1s[i].T @ x1 for i in range(n)])
print([x1.T @ Aqhs[i] @ x1 + b2s[i].T @ x1 for i in range(n)])
print(C1 @ x1 - np.ones((n+1,1)))