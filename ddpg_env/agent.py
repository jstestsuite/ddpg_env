"""DDPG (Lillicrap et al. 2015) and TD3 (Fujimoto et al. 2018) agents in PyTorch.

Actions are in [-1, 1]; the environment maps them to velocities.
"""

import copy

import numpy as np
import torch
from torch import nn


def mlp(in_dim, out_dim, hidden, final_init=3e-3):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    out = nn.Linear(last, out_dim)
    # Small output init keeps the actor's tanh and the critic's Q near zero at the start,
    # so the policy doesn't begin saturated (that was what pinned the old sigmoid speed at 0).
    nn.init.uniform_(out.weight, -final_init, final_init)
    nn.init.uniform_(out.bias, -final_init, final_init)
    layers.append(out)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim, action_dim, hidden)

    def forward(self, state):
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        # 1.0 when the transition ended in a true terminal (collision or goal): no bootstrap
        self.terminals = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminal):
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminals[i] = float(terminal)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(a[idx], device=device)
            for a in (self.states, self.actions, self.rewards, self.next_states, self.terminals)
        )


class Agent:
    """DDPG, or TD3 (Fujimoto et al. 2018) with algo='td3'. TD3 adds:
      - twin critics; the TD target uses the smaller of the two (curbs Q overestimation)
      - target policy smoothing: clipped noise on the target action
      - delayed actor and target updates (every `policy_delay` critic updates)
    """

    def __init__(self, state_dim, action_dim, algo='ddpg', hidden=(256, 256), actor_lr=1e-4,
                 critic_lr=1e-3, gamma=0.99, tau=0.005, policy_delay=2, target_noise=0.2,
                 target_noise_clip=0.5, device='cpu'):
        if algo not in ('ddpg', 'td3'):
            raise ValueError(f"algo must be 'ddpg' or 'td3', got {algo!r}")
        self.algo = algo
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.policy_delay = policy_delay if algo == 'td3' else 1
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self._updates = 0
        self._last_actor_loss = float('nan')

        self.actor = Actor(state_dim, action_dim, hidden).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden).to(self.device) if algo == 'td3' else None
        # Targets start as exact copies of the main networks
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.critic2_target = (copy.deepcopy(self.critic2).requires_grad_(False)
                               if self.critic2 is not None else None)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_params = list(self.critic.parameters())
        if self.critic2 is not None:
            critic_params += list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)

    @torch.no_grad()
    def act(self, state, noise_sigma=0.0):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(s).squeeze(0).cpu().numpy()
        if noise_sigma > 0:
            action = action + np.random.normal(0.0, noise_sigma, size=self.action_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, buffer, batch_size):
        states, actions, rewards, next_states, terminals = buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            if self.algo == 'td3':
                noise = (torch.randn_like(next_actions) * self.target_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            next_q = self.critic_target(next_states, next_actions)
            if self.algo == 'td3':
                next_q = torch.min(next_q, self.critic2_target(next_states, next_actions))
            target_q = rewards + self.gamma * (1.0 - terminals) * next_q

        q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q, target_q)
        if self.algo == 'td3':
            critic_loss = critic_loss + nn.functional.mse_loss(
                self.critic2(states, actions), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self._updates += 1
        if self._updates % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._last_actor_loss = actor_loss.item()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            if self.algo == 'td3':
                self._soft_update(self.critic2, self.critic2_target)
        return critic_loss.item(), self._last_actor_loss, q.mean().item()

    @torch.no_grad()
    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.lerp_(p, self.tau)

    def save(self, path, extra=None):
        ckpt = {
            'algo': self.algo,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'updates': self._updates,
            'extra': extra or {},
        }
        if self.algo == 'td3':
            ckpt['critic2'] = self.critic2.state_dict()
            ckpt['critic2_target'] = self.critic2_target.state_dict()
        torch.save(ckpt, path)

    def load(self, path, actor_only=False):
        """actor_only (for testing a policy) works across algorithms; resuming training
        needs the checkpoint's algo to match."""
        ckpt = torch.load(path, map_location=self.device)
        ckpt_algo = ckpt.get('algo', 'ddpg')
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        if not actor_only:
            if ckpt_algo != self.algo:
                raise ValueError(f'{path} was trained with {ckpt_algo}; '
                                 f'resume it with -p algo:={ckpt_algo}')
            self.critic.load_state_dict(ckpt['critic'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            if self.algo == 'td3':
                self.critic2.load_state_dict(ckpt['critic2'])
                self.critic2_target.load_state_dict(ckpt['critic2_target'])
            self.actor_opt.load_state_dict(ckpt['actor_opt'])
            self.critic_opt.load_state_dict(ckpt['critic_opt'])
            self._updates = ckpt.get('updates', 0)
        return ckpt.get('extra', {})
