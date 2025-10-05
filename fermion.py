
from openfermion.ops import FermionOperator
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

class Fermion_Model:
    def __init__(self,tunneling = 1.0,coulomb = 4.0):
        self.tunneling = tunneling
        self.coulomb = coulomb
        self.H=self.get_target_hamiltonian(tunneling,coulomb)

    def get_target_hamiltonian(self,tunneling = 1.0,coulomb = 4.0):
        # define the target Hamiltonian
        
        hamiltonian = fermi_hubbard(2, 2, tunneling, coulomb, periodic=False)
        H_qubit = jordan_wigner(hamiltonian)
        return H_qubit
    def evolve_state(self, psi0,t_total, hamiltonian):
        # evolve the state under the Hamiltonian for a given time
        H_sparse=get_sparse_operator(hamiltonian)
        psi_t = expm_multiply((-1j * H_sparse), psi0, start=0, stop=t_total, num=2)[-1]
        return psi_t
    
class Analog_Fermion(Fermion_Model):
    def __init__(self,model):
        self.H=self.get_target_hamiltonian(model.tunneling+0.1, model.coulomb)

if __name__ == "__main__":
    exact_model = Fermion_Model(tunneling=1.0, coulomb=4.0)
    analog_model = Analog_Fermion(exact_model)

    initial_state = Statevector.from_label('01100110').data  # Example initial state
    time=np.linspace(0,10,101)
    exact_state=initial_state
    analog_state=initial_state
    time_step=0.1
    error=[]
    for _ in time:
        analog_state = analog_model.evolve_state(exact_state, time_step, analog_model.H)
        exact_state = exact_model.evolve_state(exact_state, time_step, exact_model.H)
        error.append(np.linalg.norm(exact_state - analog_state))
    
    plt.plot(time, error)
    plt.savefig('Figures/fermion_error.pdf', bbox_inches='tight',dpi=600)
                                                  
        