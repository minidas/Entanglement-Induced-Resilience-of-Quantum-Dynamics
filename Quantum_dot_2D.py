from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
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

    def __init__(self, Omega, N=12):
        self.Omega = Omega
        self.N = N
        self.H = self.get_hamiltonian(Omega, N)

    def get_hamiltonian(self, Omega, N):
        IX = qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(N)]
        )
        return Omega * 0.5* IX
    
from random_PS import *
num_rand_PS=2**(N-2)#number of random Pauli strings

from scipy.linalg import eig
from qiskit.quantum_info import Statevector

def eigenvector_corresponding_to_maximal_eigenvalue(matrix):
    eigenvalues, eigenvectors = eig(matrix)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:,max_eigenvalue_index]

from math import sin, cos, tan

class analog_QD(Quantum_dot):
    def __init__(self, exact_model, J,  DeltaE):
        self.Omega = exact_model.Omega
        self.J = J
        self.theta = np.arctan(J/2/DeltaE)
        self.DeltaE = DeltaE
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
                  + 0.25 * J * ZZ 
                  + XZ * qt.coefficient(self.time_dependent_coefficient_XZ)
                  + YZ * qt.coefficient(self.time_dependent_coefficient_YZ))

    def time_dependent_coefficient_XZ(self,t):
        return 0.5*self.theta*self.Omega*cos(self.DeltaE*t)
    def time_dependent_coefficient_YZ(self,t):
        return -0.5*self.theta*self.Omega*sin(self.DeltaE*t)