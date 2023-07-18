# Fitted Q-iteration with Neural Network (NFQ)
# https://colab.research.google.com/drive/1dkdHP-58pUU3UG9Pl87-l54fYmukPi3c#scrollTo=PkazAOUd83l6

import numpy as np
import torch


def q_backup_sparse(env, q_values, discount=0.99):
  dS = env.num_states
  dA = env.num_actions

  new_q_values = np.zeros_like(q_values)
  value = np.max(q_values, axis=1)
  for s in range(dS):
    for a in range(dA):
      new_q_value = 0
      for ns, prob in env.transitions(s, a).items():
        new_q_value += prob * (env.reward(s,a,ns) + discount*value[ns])
      new_q_values[s,a] = new_q_value
  return new_q_values


def project_qvalues(q_values, network, optimizer, num_steps=50, weights=None):
    # regress onto q_values (aka projection)
    q_values_tensor = torch.tensor(q_values, dtype=torch.float32)
    for _ in range(num_steps):
       # Eval the network at each state
      pred_qvalues = network(torch.arange(q_values.shape[0]))
      if weights is None:
        loss = torch.mean((pred_qvalues - q_values_tensor)**2)
      else:
        loss = torch.mean(weights*(pred_qvalues - q_values_tensor)**2)
      network.zero_grad()
      loss.backward()
      optimizer.step()
    return pred_qvalues.detach().numpy()


"""
  Runs Fitted Q-iteration.

  Args:
    env: A GridEnv object.
    num_itrs (int): Number of FQI iterations to run.
    project_steps (int): Number of gradient steps used for projection.
    render (bool): If True, will plot q-values after each iteration.
  """


def fitted_q_iteration(env,
                       network,
                       num_itrs=100,
                       project_steps=50,
                       render=False,
                       weights=None,
                       **kwargs):

  dS = env.num_states
  dA = env.num_actions

  optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
  weights_tensor = None
  if weights is not None:
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

  q_values = np.zeros((dS, dA))
  for i in range(num_itrs):
    target_values = q_backup_sparse(env, q_values, **kwargs)
    q_values = project_qvalues(target_values, network, optimizer,
                               weights=weights_tensor,
                               num_steps=project_steps)
  return q_values