from qiskit.quantum_info import partial_trace,Statevector
import numpy as np

N=10

def density_matrix(state):
    return np.outer(state,state.conj())

def von_neumann_entropy(rho):
    # 计算密度矩阵的特征值
    eigenvalues = np.linalg.eigvalsh(rho)
    # 过滤掉接近零的特征值以避免数值问题
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues))
    return entropy

def RDM_entropy(state,k):
    entropy_list=[]
    for i in range(k):
        rho=partial_trace(Statevector(state),range(i+1,N))
        entropy_list.append(von_neumann_entropy(rho))
    return entropy_list

def plot_entropy(state,k):
    import matplotlib.pyplot as plt
    
