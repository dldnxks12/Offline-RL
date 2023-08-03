import numpy as np
import sys
import matplotlib.pyplot as plt

TBC_random        = np.load('TD3_BC_scott_halfcheetah-random-v0_0.npy')
TBC_medium        = np.load('TD3_BC_scott_halfcheetah-medium-v0_0.npy')
TBC_expert        = np.load('TD3_BC_scott_halfcheetah-expert-v0_0.npy')

x = np.arange(0, len(TBC_random))

plt.figure()
plt.plot(x, TBC_random, label = 'TBC_random')
plt.plot(x, TBC_medium, label = 'TBC_medium')
plt.plot(x, TBC_expert, label = 'TBC_expert')
plt.legend()
plt.xlabel('episode')
plt.ylabel('total return')
plt.title('halfcheetah-v0 TD3+BC')
plt.show()