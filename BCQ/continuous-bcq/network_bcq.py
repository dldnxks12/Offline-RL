import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device   = device
        self.discount = 0.99  # discount factor
        self.tau      = 0.005 # soft update
        self.lmbda    = 0.75  # clipped Q
        self.phi      = 0.05  # perturbate

        latent_dim = action_dim * 2

        # Perturbate network
        self.actor         = Actor(state_dim, action_dim, max_action, self.phi).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.actor_optim   = torch.optim.Adam(self.actor.parameters(), lr = 0.003)

        # Critic network
        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr = 0.003)

        # VAE network
        self.vae           = VAE(state_dim, action_dim, latent_dim, max_action, device).to(self.device)
        self.vae_optim     = torch.optim.Adam(self.vae.parameters()) # Default lr = 0.002



    def select_action(self, state): # TODO : state & action shape check
        with torch.no_grad():
            # state dim = 0에 대해 100개로 복제
            state  = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1     = self.critic.q1(state, action)
            index = q1.argmax(0)
        return action[index].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size = 100):
        for iter in range(iterations):

            state, action ,next_state, reward, not_done = replay_buffer.sample(batch_size)

            # VAE training
            deco_action, mean, std = self.vae(state, action)

            reconstruction_error = F.mse_loss(deco_action, action)
            KL_loss  = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = reconstruction_error + 0.5 * KL_loss

            self.vae_optim.zero_grad()
            vae_loss.backward()
            self.vae_optim.step()

            # Critic training
            with torch.no_grad(): # TODO : target_Q shape check
                # dim = 0을 기준으로 10번 복제
                next_state = torch.repeat_interleave(next_state, 10, 0)

                # VAE에서 perturbated sample action을 평가
                target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

                # Soft Clipped Double Q Learning
                min = torch.min(target_Q1, target_Q2)
                max = torch.max(target_Q1, target_Q2)
                target_Q = self.lmbda*min + (1 - self.lmbda)*max

                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                target_Q = reward + self.discount * target_Q * not_done

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Perturbation training
            sampled_actions     = self.vae.decode(state)
            perturbated_actions = self.actor(state, sampled_actions)

            # update with DPG
            actor_loss = -self.critic.q1(state, perturbated_actions).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()


            # Soft Update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05): # phi : perturbation
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):  # Perturbate network
        # action : VAE를 통해 sampling된 action
        x = torch.cat([state, action], 1)
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        a = self.li3(a)

        a = (self.max_action * torch.tanh(a)) * self.phi # perturbate
        return (action + a).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim+action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        Q1 = F.relu(self.l1(x))
        Q1 = F.relu(self.l2(Q1))
        Q1 = self.l3(Q1)

        Q2 = F.relu(self.l4(x))
        Q2 = F.relu(self.l5(Q2))
        Q2 = self.l6(Q2)

        return Q1, Q2

    def q1(self, state, action):
        x = torch.cat([state, action], 1)

        Q1 = F.relu(self.l1(x))
        Q1 = F.relu(self.l2(Q1))
        Q1 = self.l3(Q1)

        return Q1


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.device = device

        # Encoder
        self.ENCO1 = nn.Linear(state_dim+action_dim, 750)
        self.ENCO2 = nn.Linear(750, 750)

        self.mean    = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        # Decoder
        self.DECO1 = nn.Linear(state_dim + latent_dim, 750)
        self.DECO2 = nn.Linear(750, 750)
        self.DECO3 = nn.Linear(750, action_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        z = F.relu(self.ENCO1(x))
        z = F.relu(self.ENCO2(z))

        mean    = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std     = torch.exp(log_std)

        epsilon = torch.randn_like(std) # epsilon error
        z = mean + (std * epsilon)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z = None):

        # When sampling from VAE, the latent vactor is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        x = torch.cat([state, z], 1)

        a = F.relu(self.DECO1(x))
        a = F.relu(self.DECO2(a))
        a = self.DECO3(a)

        # [-action_size, action_size]
        a = torch.tanh(a) * self.max_action

        return a




