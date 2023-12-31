import gym
import numpy as np
import torch
import argparse
import os
import d4rl
import uuid
import json

import rem.utils as utils
import rem.DDPG as DDPG
import rem.BCQ as BCQ
import rem.TD3 as TD3
import rem.REM as REM
import rem.conv_REM as conv_REM
import rem.conv_BCQ as conv_BCQ
import rem.RSEM as RSEM
import rem.DDPG_REM as DDPG_REM


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="carla-lane-v0")              # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--buffer_type", default="Robust")             # Prepends name to filename.
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    parser.add_argument("--agent_name", default="conv_BCQ")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num_heads", default=100, type=int)
    parser.add_argument("--prefix", default="default")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    #dd4rl.set_dataset_path('/datasets')

    file_name = "%s_%s_%s_%s" % (args.agent_name, args.env_name, str(args.seed), str(args.lr))
    if args.agent_name == 'REM':
      file_name += '_%s' % (args.num_heads)
      if args.prefix != "default":
          file_name += '_%s' % (args.prefix)
    #buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    results_dir = os.path.join(args.output_dir, args.agent_name, str(uuid.uuid4()))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
        json.dump({
            'env_name': args.env_name,
            'seed': args.seed,
        }, params_file)

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    kwargs = {'lr': args.lr}
    if args.agent_name in ['conv_REM', 'REM', 'RSEM', 'DDPG_REM']:
      kwargs.update(num_heads=args.num_heads)
    if args.agent_name == 'BCQ':
      policy_agent = BCQ.BCQ
    elif args.agent_name == 'TD3':
      policy_agent = TD3.TD3
    elif args.agent_name == 'REM':
      policy_agent = REM.REM
    elif args.agent_name == 'conv_REM':
      policy_agent = conv_REM.REM
    elif args.agent_name == 'conv_BCQ':
      policy_agent = conv_BCQ.BCQ
    elif args.agent_name == 'RSEM':
      policy_agent = RSEM.RSEM
    elif args.agent_name == 'DDPG_REM':
      policy_agent = DDPG_REM.DDPG_REM
    elif args.agent_name == 'DDPG':
     policy_agent = DDPG.DDPG
     kwargs.pop('lr')
    policy = policy_agent(state_dim, action_dim, max_action, **kwargs)

    # Load buffer
    replay_buffer = utils.ReplayBuffer()
    #replay_buffer.load(buffer_name)
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

    evaluations = []

    episode_num = 0
    done = True

    training_iters = 0
    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))

        evaluations.append(evaluate_policy(policy))
        np.save(results_dir + "/" + file_name, evaluations)

        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))

