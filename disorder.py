import numpy as np
import matplotlib.pyplot as plt
from qitf_model import QITFModel, analog_QITF
import random as rand
from qiskit.quantum_info import Statevector
from tqdm import tqdm
# import pickle

num_trials = 50
time_points = [0.5 , 1 , 1.5 , 2]

exact_model = QITFModel(hx=0.809,J=1)

def plot_disorder_trails(model, time, initial_state, variance=0.01 , num_trials=num_trials, num_terms=10):
    y=[]
    for _ in tqdm(range(num_trials)):
        analog_model = analog_QITF(model, delta=[rand.normalvariate(0, variance) for _ in range(num_terms)], eta=0)
        y.append(analog_model.long_time_error(initial_state, time))
    return y

initial_state = Statevector.from_label('0'*exact_model.N).data
plots_data = []
for time in time_points:
    plots_data.append(plot_disorder_trails(exact_model, time, initial_state))

# with open('data/disorder_data_2.pkl', 'wb') as f:
#     pickle.dump(plots_data, f)
# with open('data/disorder_data.pkl', 'rb') as f:
#     plots_data = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 为每个时间点绘制曲线
colors = ["#F8A491", "#FA8BF4", "#89FACB", "#8F96FA"]
for i,t in enumerate(time_points):
<<<<<<< HEAD
    ax.plot(range(num_trials), plots_data[i], marker='o', markersize=2, zs=t, zdir='y', linestyle='')
    ax.plot(range(num_trials), [2*t*np.sqrt(exact_model.N)*0.01]*num_trials, markersize=2, zs=t, zdir='y', linestyle='dashed')
     #每个时间垂直平面绘制彩色平面
    ax.plot_surface(np.array([[0, num_trials], [0, num_trials]]), np.array([[t, t], [t, t]]), np.array([[-0.005, -0.005], [0.05,0.05]]), alpha=0.15,color="#ffff00")
=======
    ax.plot(range(num_trials), plots_data[i], marker='o', markersize=2, zs=t, zdir='y', linestyle='', color=colors[i])
    ax.plot(range(num_trials), [t*np.sqrt(exact_model.N)*0.01 for _ in range(num_trials)], zs=t, zdir='y', linestyle='--', color=colors[i], label='Error Bound')
     #每个时间垂直平面绘制彩色平面
    ax.plot_surface(np.array([[0, num_trials], [0, num_trials]]), np.array([[t, t], [t, t]]), np.array([[-0.005, -0.005], [0.22,0.22]]), alpha=0.15,color="#ffff00")
>>>>>>> 7490f2d7a384146f35930e0662f98d4972e3fa6c



# 设置标签和视角
ax.set_xlabel('Trials')
ax.set_ylabel('Evolution Time')
ax.set_zlabel('Error')
ax.set_ylim(0, 5.5)
ax.set_zlim(bottom=0)
ax.set_xlim(0, num_trials)
# ax.legend()
# ax.set_title('XXXXX')

# 调整视角以获得更好的视觉效果
ax.view_init(elev=17, azim=-35)

<<<<<<< HEAD
plt.savefig('Figures/disorder_plot_2.pdf', bbox_inches='tight', dpi=600)
# plt.show()
=======
plt.savefig('Figures/disorder_plot.pdf', bbox_inches='tight', dpi=600)
plt.show()
>>>>>>> 7490f2d7a384146f35930e0662f98d4972e3fa6c
