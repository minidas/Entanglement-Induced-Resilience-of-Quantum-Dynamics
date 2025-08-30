import numpy as np
import matplotlib.pyplot as plt
from qitf_model import QITFModel, analog_QITF
import random as rand
from qiskit.quantum_info import Statevector
from tqdm import tqdm
import pickle

num_trials = 50
time_points = [0.5 , 2 , 3.5 , 5]

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

with open('data/disorder_data.pkl', 'wb') as f:
    pickle.dump(plots_data, f)
# with open('data/disorder_data.pkl', 'rb') as f:
#     plots_data = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 为每个时间点绘制曲线
for i,t in enumerate(time_points):
    ax.plot(range(num_trials), plots_data[i], marker='o', markersize=2, zs=t, zdir='y', linestyle='')



# 设置标签和视角
ax.set_xlabel('Trials')
ax.set_ylabel('Evolution Time')
ax.set_zlabel('Error')
ax.set_title('XXXXX')

# 调整视角以获得更好的视觉效果
ax.view_init(elev=20, azim=-35)

plt.tight_layout()
plt.show()