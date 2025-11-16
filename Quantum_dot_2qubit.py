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

N=16
d=2**N

from scipy.linalg import expm

class Quantum_dot:

    def __init__(self, Omega, J0, N=16):
        self.Omega = Omega
        self.J0 = J0
        self.N = N
        self.H = self.get_hamiltonian(Omega, J0, N)

    def get_hamiltonian(self, Omega, J0, N):
        IZ = qt.tensor(
            *[qt.sigmaz() if i == 0 else qt.qeye(2) for i in range(N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 1 else qt.qeye(2) for i in range(N)]
        )
        ZZ= qt.tensor(
            *[qt.sigmaz() if i in [0,1] else qt.qeye(2) for i in range(N)]
        )
        return Omega * 0.5* IZ+ 0.25* J0 * ZZ
    
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
    def __init__(self, exact_model, h, J, DeltaE,epsilon):
        self.Omega = exact_model.Omega
        self.J0 = exact_model.J0
        self.h = h
        self.J = J
        self.theta = np.arctan(J/2/DeltaE)
        self.DeltaE = DeltaE
        self.epsilon = epsilon
        self.N = exact_model.N
        self.exact_model = exact_model

        #local pertubation
        # IX = qt.tensor(
        #     *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(self.N)]
        # )+qt.tensor(
        #     *[qt.sigmax() if i == 1 else qt.qeye(2) for i in range(self.N)]
        # )
        IZ = qt.tensor(
            *[qt.sigmaz() if i == 0 else qt.qeye(2) for i in range(self.N)]
        )
        for j in range(1,8):
            IZ += qt.tensor(
                *[qt.sigmaz() if i == j else qt.qeye(2) for i in range(self.N)]
            )

        ZZ = qt.tensor(
            *[qt.sigmaz() if i in [0,1] else qt.qeye(2) for i in range(self.N)]
        )

        for j in range(2,5):
            ZZ += qt.tensor(
                *[qt.sigmaz() if i in [0,j] else qt.qeye(2) for i in range(self.N)]
            )
        for j in range(5,8):
            ZZ += qt.tensor(
                *[qt.sigmaz() if i in [1,j] else qt.qeye(2) for i in range(self.N)]
            )

        self.H = (self.get_hamiltonian(self.Omega,self.J0,self.N) 
                  + h * IZ + 0.25 * J * ZZ)

    def time_dependent_coefficient_XZ(self,t):
        return 0.5*self.theta*self.Omega*cos(self.DeltaE*t)
    def time_dependent_coefficient_YZ(self,t):
        return -0.5*self.theta*self.Omega*sin(self.DeltaE*t)