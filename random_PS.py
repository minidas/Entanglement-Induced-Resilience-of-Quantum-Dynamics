import random as rand
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli

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