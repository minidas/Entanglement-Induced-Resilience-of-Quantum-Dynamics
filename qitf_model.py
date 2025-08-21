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
    def __init__(self, exact_model, variance):
        self.hx = exact_model.hx
        self.hy = exact_model.hy
        self.J = exact_model.J
        self.N = exact_model.N
        self.H = self.get_hamiltonian( self.hx, self.hy, self.J, self.N)
    
        # permutation_factor = rand.normalvariate(0.01, variance)
        permutation_factor = 0.01
        self.perm = permutation_factor*rand_local_PSCombin(N,N,2) # Add a permutation
        # self.perm = SparsePauliOp.from_list([('YY'+''.join(['I']*(N-2)), permutation_factor)]).to_matrix()  # Add a permutation
        self.H += self.perm

        self.U0 = self.get_evolution_segment(time_step)
        self.exact_model = exact_model

    def plot_long_time(self, initial_state, mfc, label):
        state_distance = []
        exact_state = initial_state
        analog_state = initial_state
        for i in range(num_steps):
            state_distance.append(np.linalg.norm(exact_state - analog_state))    
            exact_state = self.exact_model.U0 @ exact_state
            analog_state = self.U0 @ analog_state
        
        plt.plot(time[:-1], state_distance, color='0.3', marker='o', markersize=5, mfc=mfc, mec='k', linestyle='dashed', markeredgewidth=0.5,label=label)

    def plot_one_segment(self, initial_state, mfc, label):
        state_distance = []
        exact_state = initial_state
        analog_state = initial_state
        for i in range(num_steps):    
            analog_state = self.U0 @ exact_state
            exact_state = self.exact_model.U0 @ exact_state
            state_distance.append(np.linalg.norm(exact_state - analog_state))
        
        plt.plot(time[:-1], state_distance, color='0.3', marker='o', markersize=5, mfc=mfc, mec='k', linestyle='dashed', markeredgewidth=0.5,label=label)

def haar_random_state(n_qubits):
    dim = 2 ** n_qubits
    
    real_part = np.random.normal(0, 1, dim)
    imag_part = np.random.normal(0, 1, dim)
    state = real_part + 1j * imag_part
    
    norm = np.linalg.norm(state)
    return state / norm

if __name__ == "__main__":
    exact_model=QITFModel(hx=0.809,hy=0.9045,J=1,N=N)
    analog_model = analog_QITF(exact_model, variance=0.01)

    initial_state=Statevector.from_label('0'*N).data
    # analog_model.plot_one_segment(initial_state, mfc='#66C999', label='Separable State')
    analog_model.plot_long_time(initial_state, mfc='#66C999', label='Separable State')

    initial_state = haar_random_state(N)  # Generate a random state
    analog_model.plot_long_time(initial_state, mfc='#5DBFE9', label='Entangled State')

    linear_curve=[]
    V_fro = np.linalg.norm(analog_model.perm, 'fro')/sqrt(d)
    for t in time:
        linear_curve.append(t*V_fro)
    plt.plot(time[:-1], linear_curve[:-1], color='#397FC7', linestyle='dotted', label=r'$t\lambda\|V\|_{F}$')
    # plt.plot(time[:-1], [V_fro*time_step]*num_steps, color='#397FC7', linestyle='dotted', label=r'$\delta t\lambda\|V\|_{F}$')
    # plt.plot(time[:-1], scrambling_value,color='0.3',marker='o',markersize=5,mfc='#66C999',mec='k',linestyle='dashed',markeredgewidth=0.5)
    # plt.plot(time[:-1], simulation_error_state,color='0.3',mfc='#5DBFE9',marker='o',markersize=5,mec='k',linestyle='dashed',markeredgewidth=0.5)
    # plt.plot(time[:-1], [average_case_commutator]*num_steps,color='#66C999',linestyle='dashed')
    # plt.plot(time[:-1], [Fro_bound]*num_steps,color='#397FC7',linestyle='dotted')
    plt.xlabel('Time')
    plt.ylabel('Analog Error')
    plt.legend(loc=1)
    plt.title('QITF model, long-time evolution')
    plt.grid()
    plt.show()