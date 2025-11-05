import numpy as np
import qutip as qt
def computation_basis_state(num_qubits):
    #retrun a list of all computational basis states for num_qubits qubits, represented as integers lists
    basis_states = []
    for i in range(2**num_qubits):
        bin_str = format(i, '0' + str(num_qubits) + 'b')
        basis_states.append([int(bit) for bit in bin_str])
    return basis_states

def thermal_state(temperature,qubits):
    d=2**qubits
    basis = computation_basis_state(qubits)
    H=qt.Qobj(np.zeros((d,d)), dims=[[2]*qubits]*2)
    for i in range(qubits):
        H-=qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(qubits)])
    # psi_T = qt.Qobj(np.zeros(d), dims=[2]*N)
    psi_T = qt.Qobj(np.zeros((d,d)), dims=[[2]*qubits]*2)
    for A_subsys in basis:
        psi_T += qt.ket2dm(qt.basis([2]*qubits,A_subsys))
    # Calculate the thermal state using the Gibbs distribution
    psi_T = (-2*H * temperature).expm()*psi_T
    return psi_T/psi_T.tr()

if __name__ == "__main__":
    from entropy import von_neumann_entropy
    temp=1
    qubits=8

    from Quantum_dot_2qubit import *
    from tqdm import tqdm
    exact_model = Quantum_dot(np.pi/1e9,N=qubits)
    analog_QD = analog_QD(exact_model, h=5e5,J=1e6, DeltaE=2e8,epsilon=0.01)
    t=1e-7
    U_e=(-1j*exact_model.H*t).expm()-(-1j*analog_QD.H*t).expm()
    error_list=[]
    entropy_list=[]
    for temperature in tqdm(range(100)):
        rho_T = thermal_state((temperature)*0.008,qubits)
        error_list.append(sqrt((U_e.dag()*U_e*rho_T).tr()))
        entropy_list.append(von_neumann_entropy(rho_T.full()))

    import matplotlib.pyplot as plt
    plt.plot(entropy_list,error_list)
    plt.xlabel("Thermal entropy")
    plt.ylabel("Analog simulation error")
    plt.savefig("QD_2qubit_error_vs_entropy.pdf",bbox_inches='tight',dpi=600)