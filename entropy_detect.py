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

class QITFModel:

    def __init__(self, hx=0, hy=0.9045, J=0.4, N=10):
        self.hx = hx
        self.hy = hy
        self.J = J
        self.N = N
        self.H = self.get_hamiltonian(hx, hy, J, N)
        self.U0 = self.get_evolution_segment(time_step)

    def get_hamiltonian(self,hx,hy,J, N):
        self.X_tuples = [("X", [i], hx) for i in range(0, N)]
        self.Y_tuples = [("Y", [i], hy) for i in range(0, N)]
        self.XX_tuples = [("XX", [i, i + 1], J) for i in range(0, N - 1)]
        self.ZZ_tuples = [("ZZ", [i, i + 1], J) for i in range(0, N - 1)]
    # We create the Hamiltonian as a SparsePauliOp, via the method
    # `from_sparse_list`, and multiply by the interaction term.
        H = SparsePauliOp.from_sparse_list([*self.X_tuples, *self.Y_tuples, *self.XX_tuples], num_qubits= N)
        return H.to_matrix()
    
    def get_evolution_segment(self,segment_time=time_step):
        U0 = expm(-1j * segment_time * self.H)
        return U0
    
    def one_step_evolve(self,state):
        return self.U0 @ state

def one_step_op_evo(O,U):
    """Apply one-step operator evolution using the unitary U."""
    return U.conj().T @ O @ U

# def scrambling(oper,M):
#     commu=M@one_step_op_evo(oper,U0)- one_step_op_evo(oper,U0) @M
#     return commu@commu.conj().T

# def error_operator(operater,U):
#     return U0.conj().T @ operater @ U0 - U.conj().T @ operater @ U


import random as rand
from itertools import product

def rand_PS(N):
    # Generate a random Pauli string
    temp_tuple=''
    for i in range(N):
        temp_tuple+=rand.choice('IXYZ')

    Entangled_Op=SparsePauliOp.from_list([(temp_tuple,1)])
    O=Entangled_Op.to_matrix()
    return O

def rand_local_PS(N,k):
    characters = [''] * N
    selected_positions = rand.sample(range(N), N-k)

    for pos in selected_positions:
        characters[pos] = 'I'

# Fill the remaining positions with random choices from ['I', 'X', 'Y', 'Z']
    for i in range(N):
        if i not in selected_positions:
            characters[i] = rand.choice(['I', 'X', 'Y', 'Z'])

# Join the list into a string and print
    result = ''.join(characters)
    return SparsePauliOp.from_list([(result, 1)]).to_matrix()

def rand_state(N):
    # Generate a random state
    state=np.zeros(2**N,dtype='complex128')
    for i in range(2**N):
        state[i]=rand.uniform(-1,1)+rand.uniform(-1,1)*1j
    state=state/np.linalg.norm(state)
    return state

def rand_product_state(N):
    temp_state=''
    for i in range(N):
        temp_state+=rand.choice('01+-rl')
    product_state = Statevector.from_label(temp_state)
    return product_state.data

def normalize_matrix(O):
    # Normalize a matrix
    return O/np.linalg.norm(O,2)

def normalize_state(state):
    # Normalize a state vector
    return state/np.linalg.norm(state)

def local_PS_list(N,k):
    PS_list=[]
    for x in product('IXYZ',repeat=k):
        PS_list.append(SparsePauliOp.from_list([(''.join(x)+'I'*(N-k),1)]))
    return PS_list

def Z_op_list(N):
    Z_list=[]
    for x in product('IZ',repeat=N):
        Z_list.append(SparsePauliOp.from_list([(''.join(x),1)]))
    return Z_list


def rand_local_PSCombin(N,num_PS,support_width):
    # Generate a list of random Pauli strings
    obs=np.zeros((2**N,2**N),dtype='complex128')
    for i in range(num_PS):
        obs+=rand_local_PS(N,support_width)
    return obs

def rand_PSCombin(N,num_PS):
    obs=np.zeros((2**N,2**N),dtype='complex128')
    for i in range(num_PS):
        obs+=rand_PS(N)
    obs=obs/num_PS
    return obs

num_rand_PS=2**(N-2)#number of random Pauli strings

from scipy.linalg import eig
from qiskit.quantum_info import Statevector

def eigenvector_corresponding_to_maximal_eigenvalue(matrix):
    eigenvalues, eigenvectors = eig(matrix)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:,max_eigenvalue_index]


class analog_QITF(QITFModel):
    def __init__(self, exact_model, delta=[rand.normalvariate(0, 0.001) for _ in range(10)],eta=0.001):
        self.hx = exact_model.hx
        self.hy = exact_model.hy
        self.J = exact_model.J
        self.N = exact_model.N
        self.H = self.get_hamiltonian( self.hx, self.hy, self.J, self.N)
    
        # perturbation_factor = rand.normalvariate(0.01, variance)
        try:
            self.perm = np.zeros((2**self.N,2**self.N),dtype='complex128')
            for i, delta_k in enumerate(delta):
                # self.perm += delta_k * rand_local_PS(N,1)
                self.perm += SparsePauliOp.from_sparse_list([('X', [i],delta_k)], num_qubits=self.N).to_matrix()

        except TypeError:
            self.perm = delta * rand_local_PSCombin(N,N,1) # Add a perturbation
        
        self.perm += eta * SparsePauliOp.from_sparse_list(self.XX_tuples, num_qubits=self.N).to_matrix()  # Add an imperfection term
        
        self.H += self.perm

        self.U0 = self.get_evolution_segment(time_step)
        self.exact_model = exact_model

    def perturbation_list(self):
        perm_list=[]
        for i in range(N):
            perm_list.append(SparsePauliOp.from_sparse_list([('Z', [i],1)], num_qubits=self.N).to_matrix())
        for i in range(N-1):
            perm_list.append(SparsePauliOp.from_sparse_list([('ZZ', [i,i+1],1)], num_qubits=self.N).to_matrix())

        cross_term=[]
        for i,term in enumerate(perm_list):
            for j in range(i+1,len(perm_list)):
                cross_term.append(term.conj().T @ perm_list[j])
        return cross_term
            
    def long_time_error(self, initial_state, time):
        exact_state = self.exact_model.get_evolution_segment(time) @ initial_state
        analog_state = self.get_evolution_segment(time) @ initial_state
        return np.linalg.norm(exact_state - analog_state)

def haar_random_state(n_qubits):
    dim = 2 ** n_qubits
    
    real_part = np.random.normal(0, 1, dim)
    imag_part = np.random.normal(0, 1, dim)
    state = real_part + 1j * imag_part
    
    norm = np.linalg.norm(state)
    return state / norm

if __name__ == "__main__":
    exact_model=QITFModel(hx=0.809,hy=0.9045,J=1,N=N)
    analog_model = analog_QITF(exact_model,eta=0)

    initial_state=Statevector.from_label('0'*N).data
    evolved_state=exact_model.get_evolution_segment(final_time) @ initial_state

    error=[]
    cross=analog_model.perturbation_list()
    for term in cross:
        error.append(evolved_state.conj().T @ term @ evolved_state)

    plt.plot(range(len(error)), error,marker='o',markersize=4, mfc='#397FC7', linestyle='', label='Typical case')
    initial_state=Statevector.from_label('+'*N).data
    evolved_state=exact_model.get_evolution_segment(final_time) @ initial_state

    error=[]
    for term in cross:
        error.append(evolved_state.conj().T @ term @ evolved_state)
    plt.plot(range(len(error)), error, marker='o',markersize=4, mfc='#E74C3C', linestyle='', label='Atypical case')

    plt.xlabel('Term Label')
    plt.ylabel('Value')
    plt.legend(loc=1)
    plt.title('QIMF model, long-time evolution')
    plt.grid()
    plt.savefig('Figures/entropy_detector.pdf', bbox_inches='tight',dpi=600)