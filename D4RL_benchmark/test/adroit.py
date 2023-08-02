import gym
import d4rl


# Create the environment
env = gym.make('door-human-v0')

# dd4rl abides by the OpenAI gym interface
env.reset()

while True:
    env.step(env.action_space.sample())
    env.env.mj_render()

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use dd4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)