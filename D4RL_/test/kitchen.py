import gym
import d4rl


# Create the environment
env = gym.make('kitchen-complete-v0')

# dd4rl abides by the OpenAI gym interface
env.reset()

# visualize -> dd4rl/kitchen/kitchen_envs.py -> render modify
#while True:
    #env.step(env.action_space.sample())


# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An (N, dim_observation)-dimensional numpy array of observations
print(dataset['actions'])      # An (N, dim_action)-dimensional numpy array of actions
print(dataset['rewards'])      # An (N,)-dimensional numpy array of rewards
# Alternatively, use dd4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)