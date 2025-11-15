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

N=10
d=2**N

from scipy.linalg import expm

class Quantum_dot:

    def __init__(self, Omega, N=10):
        self.Omega = Omega
        self.N = N
        self.H = self.get_hamiltonian(Omega, N)

    def time_dependent_Omega(self,t):
        return sin(pi*t/self.T)*(self.a[0]+np.sum([self.a[k]*sin(2*pi*k*t/self.T+self.phi[k]) for k in range(1,num_rand_PS+1)]))

    def get_hamiltonian(self, Omega,a=[0.1225, 0.0672, 0.0394, -0.0297, -0.0228, 0.0040],phi=[0.0022, -0.0138, 0.0028, 0.0114, -0.0595], T=2.5e-7,N=10):
        IX = qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(N)]
        )
        self.a=a
        self.phi=phi
        self.T=T
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
        self.h = h
        self.J = J
        self.theta = np.arctan(J/2/DeltaE)
        self.DeltaE = DeltaE
        self.epsilon = epsilon
        self.N = exact_model.N
        self.exact_model = exact_model

        #local pertubation
        IX = qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(N)]
        )
        IZ = qt.tensor(
            *[qt.sigmaz() if i == 0 else qt.qeye(2) for i in range(self.N)]
        )+qt.tensor(
            *[qt.sigmaz() if i == 1 else qt.qeye(2) for i in range(self.N)]
        )

        ZZ = qt.tensor(
            *[qt.sigmaz() if i in [0,1] else qt.qeye(2) for i in range(self.N)]
        )
        XZ = qt.tensor(
            *[qt.sigmax() if i == 1 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )
        YZ = qt.tensor(
            *[qt.sigmay() if i == 1 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )
        self.H = (self.get_hamiltonian(self.Omega,self.N) 
                  + h * IZ + 0.25 * J * ZZ + epsilon*self.Omega*IX
                  + XZ * qt.coefficient(self.time_dependent_coefficient_XZ)
                  + YZ * qt.coefficient(self.time_dependent_coefficient_YZ))

    def time_dependent_coefficient_XZ(self,t):
        return 0.5*self.theta*self.Omega*cos(self.DeltaE*t)
    def time_dependent_coefficient_YZ(self,t):
        return -0.5*self.theta*self.Omega*sin(self.DeltaE*t)