from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,sin, cos, pi
from types import SimpleNamespace

if not hasattr(np.random, "mtrand"):
    np.random.mtrand = SimpleNamespace(_rand=np.random.RandomState())

import qutip as qt

final_time=6
num_steps=60
time_step=final_time/num_steps
num_rand=256

time=np.linspace(0,final_time,num_steps+1)

N=12
d=2**N

from scipy.linalg import expm

class Quantum_dot:

    def __init__(self, Omega, N=N,a=[0.2374, 0.2683, 0.1459, 0.0335, 0.0030, 0.0144],phi=[-0.0055, -0.0021, -0.0006, -0.2457, -0.0157], T=1.8e-7):
        self.Omega = float(Omega)
        self.N = N
        self.a=a
        self.phi=phi
        self.T=T
        self.H = self.get_hamiltonian(Omega, N)

    def time_dependent_Omega(self,t):
        return sin(pi*t/self.T)*(self.a[0]+np.sum([self.a[k+1]*cos(2*pi*(k+1)*t/self.T+self.phi[k]) for k in range(len(self.phi))]))

    def low_pass_filter(
        self,
        func=None,
        filter_frequency=None,
        q_factor=1 / sqrt(2),
        angular_frequency=False,
        zero_phase=True,
    ):
        """Return a low-pass-filtered callable for ``time_dependent_Omega``.

        ``time_dependent_Omega`` is a finite Fourier series, so the filter is
        applied by multiplying each harmonic by
        H(iw) = w0^2 / (w0^2 - w^2 + i*w0*w/Q).
        ``filter_frequency`` is in cycles per time unit by default; set
        ``angular_frequency=True`` for radians per time unit.
        """
        if filter_frequency is None:
            if func is None:
                raise TypeError("filter_frequency is required.")
            if callable(func):
                raise TypeError("filter_frequency is required when func is provided.")
            filter_frequency = func
            func = None

        if func is not None:
            func_name = getattr(func, "__name__", None)
            func_owner = getattr(func, "__self__", None)
            if func_name != "time_dependent_Omega" or func_owner is not self:
                raise ValueError("low_pass_filter only supports self.time_dependent_Omega.")

        filter_frequency = float(filter_frequency)
        q_factor = float(q_factor)

        if self.T <= 0:
            raise ValueError("T must be positive.")
        if filter_frequency <= 0:
            raise ValueError("filter_frequency must be positive.")
        if q_factor <= 0:
            raise ValueError("q_factor must be positive.")
        if len(self.a) < len(self.phi) + 1:
            raise ValueError("a must contain one DC-like coefficient plus one coefficient per phi.")

        cutoff_omega = filter_frequency if angular_frequency else 2 * pi * filter_frequency
        base_omega = pi / self.T
        coefficients = {1: complex(self.a[0])}

        for k, phase in enumerate(self.phi):
            n = k + 1
            coefficient = 0.5 * self.a[n] * np.exp(1j * phase)
            coefficients[2 * n + 1] = coefficients.get(2 * n + 1, 0.0) + coefficient
            coefficients[2 * n - 1] = coefficients.get(2 * n - 1, 0.0) - coefficient

        filtered_coefficients = {}
        responses = {}
        for harmonic, coefficient in coefficients.items():
            omega = harmonic * base_omega
            response = cutoff_omega**2 / (
                cutoff_omega**2 - omega**2 + 1j * cutoff_omega * omega / q_factor
            )
            responses[harmonic] = response
            filtered_coefficients[harmonic] = coefficient * (abs(response) if zero_phase else response)

        def filtered_func(t):
            t_array = np.asarray(t)
            value = np.zeros_like(t_array, dtype=float)
            for harmonic, coefficient in filtered_coefficients.items():
                value = value + np.imag(coefficient * np.exp(1j * harmonic * base_omega * t_array))
            return value.item() if np.ndim(value) == 0 else value

        filtered_func.base_omega = base_omega
        filtered_func.coefficients = coefficients
        filtered_func.filtered_coefficients = filtered_coefficients
        filtered_func.responses = responses
        filtered_func.filter_frequency = filter_frequency
        filtered_func.q_factor = q_factor
        filtered_func.angular_frequency = angular_frequency
        filtered_func.zero_phase = zero_phase
        return filtered_func

    def get_hamiltonian(self, Omega, N=N):
        IX = qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(N)]  
        )
        
        return Omega* 0.5* IX*qt.coefficient(self.time_dependent_Omega)
    
from random_PS import *
num_rand_PS=2**(N-2)#number of random Pauli strings

from scipy.linalg import eig
from qiskit.quantum_info import Statevector

def eigenvector_corresponding_to_maximal_eigenvalue(matrix):
    eigenvalues, eigenvectors = eig(matrix)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:,max_eigenvalue_index]

class analog_QD(Quantum_dot):
    def __init__(self, exact_model, h, J, DeltaE,epsilon):
        self.Omega = exact_model.Omega
        self.a=exact_model.a
        self.phi=exact_model.phi
        self.T=exact_model.T
        self.h = h
        self.J = J
        self.theta = np.arctan(J/2/DeltaE)
        self.DeltaE = DeltaE
        self.epsilon = epsilon
        self.N = exact_model.N
        self.exact_model = exact_model

        #local pertubation
        X=qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(self.N)]
        )
        Z=qt.tensor(
            *[qt.sigmaz() if i == 0 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 1 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 2 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 3 else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i == 4 else qt.qeye(2) for i in range(self.N)]
        )
        ZZ = qt.tensor(
            *[qt.sigmaz() if i in [0,1] else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i in [0,2] else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i in [0,3] else qt.qeye(2) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmaz() if i in [0,4] else qt.qeye(2) for i in range(self.N)]
        )
        XZ = qt.tensor(
            *[qt.sigmax() if i == 1 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmax() if i == 2 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmax() if i == 3 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmax() if i == 4 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )
        YZ = qt.tensor(
            *[qt.sigmay() if i == 1 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmay() if i == 2 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmay() if i == 3 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )+ qt.tensor(
            *[qt.sigmay() if i == 4 else (qt.sigmaz() if i == 0 else qt.qeye(2)) for i in range(self.N)]
        )
        self.H = (self.get_hamiltonian(self.Omega,self.N) 
                  + h * Z + 0.25 * J * ZZ + epsilon*self.Omega*X*qt.coefficient(self.time_dependent_Omega)
                  + XZ * qt.coefficient(self.time_dependent_coefficient_XZ)
                  + YZ * qt.coefficient(self.time_dependent_coefficient_YZ))
    def get_hamiltonian(self, Omega, N=N):
        IX = qt.tensor(
            *[qt.sigmax() if i == 0 else qt.qeye(2) for i in range(N)]  
        )
        
        return Omega* 0.5* IX*qt.coefficient(self.low_pass_filter(filter_frequency=35e6, q_factor=1/sqrt(2)))

    def time_dependent_coefficient_XZ(self,t):
        return 0.5*self.theta*self.Omega*cos(self.DeltaE*t)*self.time_dependent_Omega(t)
    def time_dependent_coefficient_YZ(self,t):
        return -0.5*self.theta*self.Omega*sin(self.DeltaE*t)*self.time_dependent_Omega(t)

if __name__ == "__main__":
    def plot_low_pass_filter_comparison(
        model=None,
        filter_frequency=40e6,
        q_factor=1 / sqrt(2),
        num_samples=2048,
        output_path="time_dependent_Omega_low_pass_comparison.png",
    ):
        if model is None:
            model = Quantum_dot.__new__(Quantum_dot)
            model.Omega = 1e9
            model.N = N
            model.a = [0.2374, 0.2683, 0.1459, 0.0335, 0.0030, 0.0144]
            model.phi = [-0.0055, -0.0021, -0.0006, -0.2457, -0.0157]
            model.T = 1.8e-7

        t_values = np.linspace(0.0, model.T, num_samples)
        a = np.asarray(model.a, dtype=float)
        phi = np.asarray(model.phi, dtype=float)
        harmonics = np.arange(1, phi.size + 1, dtype=float)
        carrier = a[0] + np.sum(
            a[1 : phi.size + 1, None]
            * np.cos(2 * pi * harmonics[:, None] * t_values[None, :] / model.T + phi[:, None]),
            axis=0,
        )
        original_values = model.Omega * np.sin(pi * t_values / model.T) * carrier / 1e6
        filtered_values = (
            model.Omega
            * model.low_pass_filter(filter_frequency=filter_frequency, q_factor=q_factor)(t_values)
            / 1e6
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_values * 1e9, original_values, label="Original", linewidth=2)
        ax.plot(
            t_values * 1e9,
            filtered_values,
            label=f"Low-pass ({filter_frequency / 1e6:g} MHz)",
            linewidth=2,
        )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(r"$\Omega(t)$ (MHz)")
        ax.set_title("Time-dependent Omega before and after low-pass filtering")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        return fig, ax

    plot_low_pass_filter_comparison(output_path=None)
    plt.show()
