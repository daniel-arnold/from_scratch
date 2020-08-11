"""
Conjugate Gradient algorithm to solve the system of equations Ax=b
Note that this also solves the quadratic minimization problem:
1/2 x' @ Q x + b' @ x

Steepest Descent is implemented for comparison

@author: daniel arnold
"""

#Imports
import numpy as np
import numpy.linalg as LA
from sklearn import datasets
import matplotlib.pyplot as plt

####################################################################
#Steepest Descent Solver

class SteepestDescentSolver:
    def __init__(self, A, b, n, epsilon=1e-4):
        self.A = A
        self.b = b
        self.n = n
        self.epsilon = epsilon
        print("SD - epsilon (convergence): ", self.epsilon)

    def solve(self, x0):
        num_iters = 0
        solved = False
        x = x0
        while not solved:
            #calculate residual
            r = self.b - self.A @ x
            #check convergence criteria
            if(LA.norm(r,2) <= self.epsilon):
                solved = True
                break
            #compute steepest descent step size
            alpha = ((r.T) @ r) / ((r.T) @ A @ r)
            #compute next candidate solution
            x = x + alpha * r
            num_iters += 1

        return x, num_iters

####################################################################
#Conjugate Gradient Solver

class ConjugateGradientSolver:
    def __init__(self, A, b, n, epsilon=1e-4):
        self.A = A
        self.b = b
        self.n = n
        self.epsilon = epsilon
        print("CG - epsilon (convergence): ", self.epsilon)

    def solve(self, x0):
        num_iters = 0
        solved = False
        x = x0
        r = self.b - self.A @ x0
        d = r
        while not solved:
            #calculate optimal step size for search direction, d
            alpha = ((r.T) @ r) / ((d.T) @ A @ d)
            #compute next solution candidate
            x = x + alpha * d
            #compute next residual
            r_ip1 = r - alpha * self.A @ d
            #Gram Schmidt orthogonalization constant
            Beta = (r_ip1.T) @ r_ip1 / ((r.T) @ r.T)
            #next orthogonal search direction
            d = r_ip1 + Beta * d
            #check convergence criteria
            if(LA.norm(r,2) <= self.epsilon):
                solved = True
                break

            r = r_ip1
            num_iters += 1

        return x, num_iters

########################################################################
if __name__ == "__main__":
    
    #create A and b
    print("\nProgram Running")
    n = 20
    print("dimension of x: ", n)
    print("\n")
    b_low = -2
    b_high = 2
    A = datasets.make_spd_matrix(n)
    b = np.random.uniform(b_low, b_high, n)
    x0 = np.random.uniform(b_low, b_high, n)
    
    x_star = LA.inv(A) @ b

    sdsolver = SteepestDescentSolver(A,b,n)
    x_sd, num_iters_sd = sdsolver.solve(x0)

    print("2-norm of SD difference from optimal: ", LA.norm(x_star - x_sd, 2))
    print("number of iterations: ", num_iters_sd)

    cgsolver = ConjugateGradientSolver(A,b,n)
    x_cg, num_iters_cg = cgsolver.solve(x0)

    print("2-norm of CG difference from optimal: ", LA.norm(x_star - x_cg, 2))
    print("number of iterations: ", num_iters_cg)