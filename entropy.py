from qiskit.quantum_info import partial_trace,Statevector
import numpy as np
from qitf_model import QITFModel

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

def plot_entropy(state,model : QITFModel,k,time=time,mfc='#F3A33A',label='entropy'):
    entropy_list = []
    for i in time:
        entropy_list.append(RDM_entropy(state,k)[-1])
        state = model.evolve(state)
    plt.plot(time[:-1], entropy_list[:-1], marker=mfc, label=label)