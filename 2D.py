from QIMF_2D import *
import numpy as np

# exact_model = QIMF_2D(hx=0.809,J=1)
# analog_model = analog_QITF(exact_model,eta=0)

import pickle

# with open('./data/2D_analog_model[di].pkl', 'wb') as f:
#     pickle.dump(analog_model, f)

with open('./data/2D_exact_model.pkl', 'rb') as f:
    exact_model = pickle.load(f)
with open('./data/2D_analog_model[di].pkl', 'rb') as f:
    analog_model = pickle.load(f)

initial_state = Statevector.from_label('0'*N).data  # Typical case state

fro_curve=[]
spec_curve=[]
V_fro = np.linalg.norm(analog_model.perm, 'fro')/sqrt(d)
V_spec = np.linalg.norm(analog_model.perm, 2)

estimate_fro = 0
U_int = exact_model.get_evolution_segment(segment_time=0.01)
state=initial_state
for t in np.linspace(0,10,11):
    fro_curve.append(estimate_fro)
    for i in range(10):
        estimate_fro += 0.01 * sqrt(abs(state.conj().T@(analog_model.perm.conj().T@analog_model.perm)@state))
        state = U_int @ state
for t in range(50):
    fro_curve.append(fro_curve[-1]+0.1*V_fro)

estimate_spec = 0
initial_state = Statevector.from_label('+'*N).data  # Atypical case state
state=initial_state
for t in np.linspace(0,60,61):
    spec_curve.append(estimate_spec)
    for i in range(10):
        estimate_spec += 0.01 * np.linalg.norm(analog_model.perm@state)
        state = U_int @ state

initial_state=Statevector.from_label('+'*N).data
    # analog_model.plot_one_segment(initial_state, mfc='#66C999', label='Separable State')
analog_model.plot_long_time(initial_state, mfc="#E19D92", label='Empirical (Atypical)')

# initial_state = haar_random_state(N)  # Generate a random state
initial_state = Statevector.from_label('0'*N).data  # Typical case state
analog_model.plot_long_time(initial_state, mfc='#7E8CAD', label='Empirical (Typical)')

linear_curve=[]
for t in time:
    linear_curve.append(t*V_fro)
worst_curve =[]
for t in time:
    worst_curve.append(t*V_spec)

plt.plot(time[:-1], fro_curve[:-1], color="#7E8CAD", linestyle='dotted', label='Our bound (Typical)',linewidth=3)
plt.plot(time[:-1], spec_curve[:-1], color="#E19D92", linestyle='dotted', label='Our bound (Atypical)',linewidth=3)
plt.plot(time[:-1], worst_curve[:-1], color="#EAC47C", linestyle='dotted', label=r'$t\lambda\|V\|_{\infty}$',linewidth=3)
plt.plot(time[:-1], linear_curve[:-1], color="#A5C2CD", linestyle='dotted', label=r'$t\lambda\|V\|_{F}$',linewidth=3)

plt.xlabel('Time')
plt.ylabel('Analog Error')
# plt.ylim( top=0.3)
plt.legend(loc=1)
plt.title('2D-lattice QIMF, long-time evolution with disorder & imperfections')
plt.grid()
ax1=plt.gca()

from mpl_toolkits.axes_grid1.inset_locator import mark_inset,BboxConnector

ax2=plt.axes([0.2,0.55,0.3,0.3])
ax2.set_xlim(0,1)
ax2.set_ylim(0,0.07)
ax2.set_facecolor("#ffffe6")

analog_model.plot_long_time(initial_state, mfc='#7E8CAD',label='Typical Case State')
plt.plot(time[:-1], fro_curve[:-1], color='#7E8CAD', linestyle='dotted', linewidth=3)
plt.plot(time[:-1], linear_curve[:-1], color="#A5C2CD", linestyle='dotted', linewidth=3)
initial_state = Statevector.from_label('+'*N).data  # Atypical case state
analog_model.plot_long_time(initial_state, mfc="#E19D92", label='Atypical Case State')
plt.plot(time[:-1], spec_curve[:-1], color="#E19D92", linestyle='dotted', label='Our bound (Atypical)',linewidth=3)
plt.plot(time[:-1], worst_curve[:-1], color="#EAC47C", linestyle='dotted', label=r'$t\lambda\|V\|_{\infty}$',linewidth=3)

plt.grid()
# plt.tight_layout()
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="#ffffda", ec="0.5",linestyle='dashed')
plt.savefig('./Figures/long-time[di,2D].pdf', bbox_inches='tight',dpi=600)
