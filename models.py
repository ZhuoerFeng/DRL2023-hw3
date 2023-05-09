import torch
import torch.nn as nn
from utils import mlp


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, action_space, hidden_dims=[400, 300], output_activation=nn.Tanh):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.fcs = mlp(num_states, hidden_dims, num_actions, output_activation=output_activation)

    def _normalize(self, action) -> torch.Tensor:
        """
        Normalize the action value to the action space range.
        Hint: the return values of self.fcs is between -1 and 1 since we use tanh as output activation, while we want the action ranges to be (self.action_space.low, self.action_space.high). You can normalize the action value to the action space range linearly.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        return (action + 1) / 2. * (self.action_space.high - self.action_space.low) + self.action_space.low
        ############################

    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        return super().to(device)

    def forward(self, x):
        # use tanh as output activation
        return self._normalize(self.fcs(x))


class SoftActor(Actor):
    def __init__(self, num_states, num_actions, hidden_size, action_space, log_std_min=-20, log_std_max=2):
        super().__init__(num_states, num_actions * 2, action_space, hidden_dims=hidden_size, output_activation=nn.Identity)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain mean and log(std) from fully-connected network.
        Limit the value of log_std to the specified range.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        mean_std = self.fcs(state) # output activation is identity
        mean, log_std = torch.chunk(mean_std, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        ############################
        return mean, log_std

    def evaluate(self, state, sample=True) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        if not sample:
            return self._normalize(torch.tanh(mean)), None

        # sample action from N(mean, std) if sample is True
        # obtain log_prob for policy and Q function update
        # Hint: remember the reparameterization trick, and perform tanh normalization
        # This library might be helpful: torch.distributions
        ############################
        # YOUR IMPLEMENTATION HERE #
        m = torch.distributions.normal.Normal(loc=mean, scale=log_std.exp())
        _action = m.rsample()
        action = torch.tanh(_action)
        # scale = (self.action_space.high - self.action_space.low) / 2 
        log_prob = m.log_prob(_action) - torch.log((1 - action.pow(2)) + 1e-6)
        ############################
        return self._normalize(action), log_prob


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims):
        super().__init__()
        self.fcs = mlp(num_states + num_actions, hidden_dims, 1)

    def forward(self, state, action):
        return self.fcs(torch.cat([state, action], dim=1)).squeeze()
