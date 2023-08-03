import numpy as np
import sys
import matplotlib.pyplot as plt

# Behavioral : DDPG
behavioral = np.load('behavioral_halfcheetah-random-v0_0.npy')

# BCQ
BCQ_random        = np.load('BCQ_halfcheetah-random-v0_0.npy')
BCQ_medium        = np.load('BCQ_halfcheetah-medium-v0_0.npy')
BCQ_expert        = np.load('BCQ_halfcheetah-expert-v0_0.npy')

# TD3 + BC
TBC_random        = np.load('TD3_BC_scott_halfcheetah-random-v0_0.npy')
TBC_medium        = np.load('TD3_BC_scott_halfcheetah-medium-v0_0.npy')
TBC_expert        = np.load('TD3_BC_scott_halfcheetah-expert-v0_0.npy')

x = np.arange(0, len(TBC_random))

plt.figure()
plt.plot(x, behavioral, label = 'Behavioral')
plt.plot(x, BCQ_random, label = 'BCQ_random')
plt.plot(x, TBC_random, label = 'TBC_random')
plt.legend()
plt.xlabel('episode')
plt.ylabel('total return')
plt.title('halfcheetah-v0')

plt.figure()
plt.plot(x, behavioral, label = 'Behavioral')
plt.plot(x, BCQ_medium, label = 'BCQ_medium')
plt.plot(x, TBC_medium, label = 'TBC_medium')
plt.legend()
plt.xlabel('episode')
plt.ylabel('total return')
plt.title('halfcheetah-v0')

plt.figure()
plt.plot(x, behavioral, label = 'Behavioral')
plt.plot(x, BCQ_expert, label = 'BCQ_expert')
plt.plot(x, TBC_expert, label = 'TBC_expert')
plt.legend()
plt.xlabel('episode')
plt.ylabel('total return')
plt.title('halfcheetah-v0')

plt.show()