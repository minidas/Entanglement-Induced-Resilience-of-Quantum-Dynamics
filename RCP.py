from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,sin, cos, tan, pi
import qutip as qt

final_time=6
num_steps=60
time_step=final_time/num_steps
num_rand=256

time=np.linspace(0,final_time,num_steps+1)

N=12
d=2**N

from scipy.linalg import expm

class Quantum_dot:

    def __init__(self, Omega, N=N,a=[0.2374, 0.2683, 0.1459, 0.0335, 0.0030, 0.0144],phi=[-0.0055, -0.0021, -0.0006, -0.2457, -0.0157], T=1.8e-7):
        self.Omega = float(Omega)
        self.N = N
        self.a=a
        self.phi=phi
        self.T=T
        self.H = self.get_hamiltonian(Omega, N)

    def time_dependent_Omega(self,t):
        return sin(pi*t/self.T)*(self.a[0]+np.sum([self.a[k+1]*cos(2*pi*(k+1)*t/self.T+self.phi[k]) for k in range(len(self.phi))]))

    def get_hamiltonian(self, Omega, N=N):
        IX = qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(N)]  
        )
        
        return Omega* 0.5* IX*qt.coefficient(self.time_dependent_Omega)
    
from random_PS import *
num_rand_PS=2**(N-2)#number of random Pauli strings

from scipy.linalg import eig
from qiskit.quantum_info import Statevector

def eigenvector_corresponding_to_maximal_eigenvalue(matrix):
    eigenvalues, eigenvectors = eig(matrix)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:,max_eigenvalue_index]

class analog_QD(Quantum_dot):
    def __init__(self, exact_model, h, J, DeltaE,epsilon):
        self.Omega = exact_model.Omega
        self.a=exact_model.a
        self.phi=exact_model.phi
        self.T=exact_model.T
        self.h = h
        self.J = J
        self.theta = np.arctan(J/2/DeltaE)
        self.DeltaE = DeltaE
        self.epsilon = epsilon
        self.N = exact_model.N
        self.exact_model = exact_model

        #local pertubation
        X=qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(self.N)]
        )
        Z=qt.tensor(
            *[qt.sigmaz() if i == 0 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 1 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 2 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 3 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 4 else qt.qeye(2) for i in range(self.N)]
        )
        ZZ = qt.tensor(
            *[qt.sigmaz() if i in [0,1] else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i in [0,2] else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i in [0,3] else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i in [0,4] else qt.qeye(2) for i in range(self.N)]
        )
        XZ = qt.tensor(
            *[qt.sigmax() if i == 1 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmax() if i == 2 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmax() if i == 3 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmax() if i == 4 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )
        YZ = qt.tensor(
            *[qt.sigmay() if i == 1 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmay() if i == 2 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmay() if i == 3 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmay() if i == 4 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )
        self.H = (self.get_hamiltonian(self.Omega,self.N) 
                  + h * Z + 0.25 * J * ZZ + epsilon*self.Omega*X*qt.coefficient(self.time_dependent_Omega)
                #   + 0.25 * J * ZZ
                  + XZ * qt.coefficient(self.time_dependent_coefficient_XZ)
                  + YZ * qt.coefficient(self.time_dependent_coefficient_YZ))

    def time_dependent_coefficient_XZ(self,t):
        return 0.5*self.theta*self.Omega*cos(self.DeltaE*t)*self.time_dependent_Omega(t)
    def time_dependent_coefficient_YZ(self,t):
        return -0.5*self.theta*self.Omega*sin(self.DeltaE*t)*self.time_dependent_Omega(t)