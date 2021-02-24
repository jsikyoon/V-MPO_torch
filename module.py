import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical

class Memory:
    def __init__(self):
        self.ts = []
        self.actions = []
        self.rep_states = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.ts[:]
        del self.actions[:]
        del self.rep_states[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class StateRepresentation(nn.Module):
    def __init__(self, state_rep, state_dim, action_dim, n_latent_var, device):
        super(StateRepresentation, self).__init__()

        self.state_rep = state_rep
        self.action_dim = action_dim
        self.device = device

        inp_dim = state_dim + action_dim + 1 # current state, previous action and reward
        if state_rep == 'none':
            out_dim = state_dim
        else:
            out_dim = n_latent_var

        if state_rep == 'lstm':
            self.layer = nn.LSTMCell(inp_dim, out_dim)
            self.h0 = nn.Parameter(torch.rand(n_latent_var))
            self.c0 = nn.Parameter(torch.rand(n_latent_var))
        elif state_rep == 'trxl':
            raise NotImplemented
        elif state_rep == 'gtrxl':
            raise NotImplemented

        self.init_action = nn.Parameter(torch.rand(action_dim))
        self.init_reward = nn.Parameter(torch.rand(1))

    def forward(self, t, state, _prev_action=None, _prev_reward=None):

        state = torch.from_numpy(state).float().to(self.device)

        if self.state_rep == 'none':
                return state

        if t==0:
            prev_action = self.init_action
            prev_reward = self.init_reward
            if self.state_rep == 'lstm':
                self.h = self.h0.unsqueeze(0)
                self.c = self.c0.unsqueeze(0)
        else:
            prev_action = torch.zeros(self.action_dim).to(self.device)
            prev_action[_prev_action] = 1
            prev_reward = torch.from_numpy(np.array([_prev_reward])).float().to(self.device)

        # [1, inp_dim]
        inp = torch.cat([state, prev_action, prev_reward], dim=0)
        inp = inp.unsqueeze(0)

        if self.state_rep == 'lstm':
            h, c = self.layer(inp, (self.h, self.c))
            self.h = h
            self.c = c
            return h[0]
        elif self.state_rep == 'trxl':
            raise NotImplemented
        elif self.state_rep == 'gtrxl':
            raise NotImplemented

    def batch_forward(self, ts, states, actions, rewards):

        if self.state_rep == 'none':
            rep_states = torch.from_numpy(np.array(states)).float().to(self.device)
        elif self.state_rep == 'lstm':
            rep_states = []
            for i in range(len(ts)):
                t = ts[i]
                state = states[i]
                action = actions[i]
                reward = rewards[i]

                if i == 0:
                    prev_action = None
                    prev_reward = None
                else:
                    prev_action = actions[i-1]
                    prev_reward = rewards[i-1]

                if t==0:
                    rep_states.append(self.forward(t, state))
                else:
                    rep_states.append(self.forward(t, state, prev_action, prev_reward))

            rep_states = torch.stack(rep_states, dim=0)
        elif self.state_rep == 'trxl':
            raise NotImplemented
        elif self.state_rep == 'gtrxl':
            raise NotImplemented

        return rep_states



class ActorCritic(nn.Module):
    def __init__(self, model, state_dim, action_dim, n_latent_var, state_rep, device):
        super(ActorCritic, self).__init__()

        self.model = model
        self.device = device
        self.state_rep = state_rep
        if state_rep == 'none':
            inp_dim = state_dim
        else:
            inp_dim = n_latent_var

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(inp_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(inp_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
            )

        # shared state representation module
        self.shared_layer = StateRepresentation(state_rep, state_dim, action_dim, n_latent_var, device)

    def forward(self):
        raise NotImplementedError

    def act(self, t, state, memory):

        if t==0:
            rep_state = self.shared_layer(t, state)
        else:
            rep_state = self.shared_layer(t, state, memory.actions[-1], memory.rewards[-1])

        action_probs = self.action_layer(rep_state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.ts.append(t)
        memory.states.append(state)
        memory.rep_states.append(rep_state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, ts, states, actions, rewards):

        rep_states = self.shared_layer.batch_forward(ts, states, actions, rewards)

        action_probs = self.action_layer(rep_states.detach())
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        if self.model == 'ppo':
            dist_entropy = dist.entropy()
        elif self.model == 'vmpo':
            dist_probs = dist.probs

        state_value = self.value_layer(rep_states)

        if self.model == 'ppo':
            return action_logprobs, torch.squeeze(state_value), dist_entropy
        elif self.model == 'vmpo':
            return action_logprobs, torch.squeeze(state_value), dist_probs

class VMPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, state_rep, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True)
        self.eps_eta = 0.02
        self.eps_alpha = 0.1
        self.device = device

        self.policy = ActorCritic('vmpo', state_dim, action_dim, n_latent_var, state_rep, device).to(device)

        params = [
                {'params': self.policy.parameters()},
                {'params': self.eta},
                {'params': self.alpha}
            ]

        self.optimizer = torch.optim.Adam(params, lr=lr, betas=betas)
        self.policy_old = ActorCritic('vmpo', state_dim, action_dim, n_latent_var, state_rep, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def get_KL(self, prob1, logprob1, logprob2):
        kl = prob1 * (logprob1 - logprob2)
        return kl.sum(1, keepdim=True)

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_ts = memory.ts
        old_states = memory.states
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_rewards = memory.rewards

        # Get old probs and old advantages
        with torch.no_grad():
            _, old_state_values, old_dist_probs = self.policy_old.evaluate(old_ts, old_states, old_actions, old_rewards)
            advantages = rewards - old_state_values.detach()

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating sampled actions and values:
            logprobs, state_values, dist_probs = self.policy.evaluate(old_ts, old_states, old_actions, old_rewards)

            # Get samples with top half advantages
            advprobs = torch.stack((advantages, logprobs))
            advprobs = advprobs[:, torch.sort(advprobs[0], descending=True).indices]
            good_advantages = advprobs[0, :len(old_states)//2]
            good_logprobs = advprobs[1, :len(old_states)//2]

            # Get losses
            phis = torch.exp(good_advantages/self.eta.detach())/torch.sum(torch.exp(good_advantages/self.eta.detach()))
            L_pi = -phis*good_logprobs
            L_eta = self.eta*self.eps_eta+self.eta*torch.log(torch.mean(torch.exp(good_advantages/self.eta)))

            KL = self.get_KL(old_dist_probs.detach(), torch.log(old_dist_probs).detach(), torch.log(dist_probs))

            L_alpha = torch.mean(self.alpha*(self.eps_alpha-KL.detach())+self.alpha.detach()*KL)

            loss = L_pi + L_eta + L_alpha + 0.5*self.MseLoss(state_values, rewards)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            with torch.no_grad():
                self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))

            #if i == self.K_epochs - 1:
            #    print(torch.mean(KL).item(), self.alpha.item())

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, state_rep, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic('ppo', state_dim, action_dim, n_latent_var, state_rep, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic('ppo', state_dim, action_dim, n_latent_var, state_rep, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_ts = memory.ts
        old_states = memory.states
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_rewards = memory.rewards
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Get old probs and old advantages
        with torch.no_grad():
            _, old_state_values, old_dist_probs = self.policy_old.evaluate(old_ts, old_states, old_actions, old_rewards)
            advantages = rewards - old_state_values.detach()

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating sampled actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_ts, old_states, old_actions, old_rewards)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).unsqueeze(-1) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
