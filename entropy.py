from qiskit.quantum_info import partial_trace,Statevector
import numpy as np
from qitf_model import QITFModel
import matplotlib.pyplot as plt


N=10
final_time=6
num_steps=60
time_step=final_time/num_steps
num_rand=256

time=np.linspace(0,final_time,num_steps+1)

def density_matrix(state):
    return np.outer(state,state.conj())

def von_neumann_entropy(rho):
    # 计算密度矩阵的特征值
    eigenvalues = np.linalg.eigvalsh(rho)
    # 过滤掉接近零的特征值以避免数值问题
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues)/np.log(2))
    return entropy

def RDM_entropy(state,k,qubits=N):
    entropy_list=[]
    for i in range(k):
        rho=partial_trace(Statevector(state),range(i+1,qubits))
        entropy_list.append(von_neumann_entropy(rho))
    return entropy_list

def entropy_data(state,model : QITFModel,k,time=time,qubits=N):
    entropy_list = []
    try:
        for i in time:
            entropy_list.append(RDM_entropy(state,k,qubits)[-1])
            state = model.one_step_evolve(state)
        return entropy_list
    except TypeError:
        for i in time:
            rho=partial_trace(Statevector(state),k)
            entropy_list.append(von_neumann_entropy(rho))
            state = model.one_step_evolve(state)
        return entropy_list

def plot_entropy(state,model : QITFModel,k,time=time,mfc='#F3A33A',label='entropy'):
    entropy_list = entropy_data(state, model, k, time)
    plt.plot(time[:-1], entropy_list[:-1], color=mfc, label=label,linestyle='dashed')