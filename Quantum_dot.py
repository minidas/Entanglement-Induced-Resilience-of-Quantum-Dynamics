from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

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
        self.U0 = self.get_evolution_segment(time_step)

    def get_hamiltonian(self, Omega, N):
        self.drive_tuple = [("X", [1], Omega*0.5)]
        H = SparsePauliOp.from_sparse_list(self.drive_tuple, num_qubits= N)
        return H.to_matrix()
    
    def get_evolution_segment(self,segment_time=time_step):
        U0 = expm(-1j * segment_time * self.H)
        return U0
    
    def one_step_evolve(self,state):
        return self.U0 @ state

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
    def __init__(self, exact_model, J, theta, DeltaE):
        self.Omega = exact_model.Omega
        self.J = J
        self.theta = theta
        self.DeltaE = DeltaE
        self.N = exact_model.N
        self.H = self.get_hamiltonian( self.Omega,self.N)

        self.perm1 = SparsePauliOp.from_sparse_list([( "ZZ" , [0,1] ,J * 0.25 )],num_qubits=self.N).to_matrix()  # Add a permutation

        self.exact_model = exact_model
        
    def time_dependent_term(self, time):
        self.perm2 = 0.5 * self.Omega * tan(self.theta) * SparsePauliOp.from_sparse_list([( "XZ" , [0,1], cos(self.DeltaE*time)),( "YZ" , [0,1], -sin(self.DeltaE*time))],num_qubits=self.N).to_matrix()
        self.perm = self.perm1 + self.perm2
        U0 = expm(-1j * 0.1 * time_step * (self.H + self.perm))
        return U0 , self.perm


def haar_random_state(n_qubits):
    dim = 2 ** n_qubits
    
    real_part = np.random.normal(0, 1, dim)
    imag_part = np.random.normal(0, 1, dim)
    state = real_part + 1j * imag_part
    
    norm = np.linalg.norm(state)
    return state / norm
