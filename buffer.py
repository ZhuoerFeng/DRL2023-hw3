import torch
import numpy as np
from collections import deque


def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    if not cfg.use_per:
        if cfg.nstep == 1:
            return ReplayBuffer(cfg.capacity, **args)
        else:
            return NStepReplayBuffer(cfg.capacity, cfg.nstep, cfg.gamma, **args)
    else:
        if cfg.nstep == 1:
            return PrioritizedReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, **args)
        else:
            return PrioritizedNStepReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, cfg.nstep, cfg.gamma, **args)


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size, device, seed):
        self.device = device
        self.state = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.action = torch.zeros(capacity, action_size, dtype=torch.float).contiguous().pin_memory()
        self.reward = torch.zeros(capacity, dtype=torch.float).contiguous().pin_memory()
        self.next_state = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.done = torch.zeros(capacity, dtype=torch.int).contiguous().pin_memory()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer and update the index and size of the buffer
        # you may need to convert the data type to torch.tensor

        ############################
        # YOUR IMPLEMENTATION HERE #
        if self.size < self.capacity:
            self.state[self.size] = torch.as_tensor(state)
            self.action[self.size] = torch.as_tensor(action)
            self.reward[self.size] = torch.as_tensor(reward) # if done != 0 else 0)
            self.next_state[self.size] = torch.as_tensor(next_state)
            self.done[self.size] = torch.as_tensor(done)
            self.size += 1
            self.idx = self.size
        else:
            if self.idx >= self.size:
                self.idx -= self.size
            self.state[self.idx] = torch.as_tensor(state)
            self.action[self.idx] = torch.as_tensor(action)
            self.reward[self.idx] = torch.as_tensor(reward) # if done != 0 else 0)
            self.next_state[self.idx] = torch.as_tensor(next_state)
            self.done[self.idx] = torch.as_tensor(done)
            self.idx += 1

        ############################

    def sample(self, batch_size):
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = ()
        # get a batch of data from the buffer according to the sample_idxs
        # please transfer the data to the corresponding device before return
        ############################
        # YOUR IMPLEMENTATION HERE #

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
        )

        ############################
        return batch


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, action_size, device, seed):
        super().__init__(capacity, state_size, action_size, device, seed)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done forwards, break if there's a done"""
        ############################
        # （OPTIONAL) YOUR IMPLEMENTATION HERE #
        discount = 1
        terminated = False
        state, action, _, _ = self.n_step_buffer[0]
        reward = 0
        for sard in self.n_step_buffer:
            _, _, r, d = sard
            if not terminated:
                reward += r * discount
            discount *= self.gamma
            if d:
                terminated = True
        done = 1 if terminated else 0
        ############################
        return state, action, reward, done

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, action_size, device, seed):
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, action_size, device, seed)

    def add(self, transition):
        """
        Add a new experience to memory, and update it's priority to the max_priority.
        """
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #
        state, action, reward, next_state, done = transition
        if self.size < self.capacity:
            self.size += 1
        else:
            if self.idx >= self.capacity:
                self.idx -= self.capacity
            assert self.idx < self.size, "Buffer idx >= size"
        self.state[self.idx] = torch.as_tensor(state)
        self.action[self.idx] = torch.as_tensor(action)
        self.reward[self.idx] = torch.as_tensor(reward)
        self.next_state[self.idx] = torch.as_tensor(next_state)
        self.done[self.idx] = torch.as_tensor(done)
        self.priorities[self.idx] = self.max_priority
        self.idx += 1
        ############################

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with priority, and calculates the weights used for the correction of bias used in the Q-learning update
        Returns:
            batch: a batch of experiences as in the normal replay buffer
            weights: torch.Tensor (batch_size, ), importance sampling weights for each sample
            sample_idxs: numpy.ndarray (batch_size, ), the indexes of the sample in the buffer
        """
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #
        p_i = self.priorities[:self.size]
        p_i = p_i / np.sum(p_i)
        sample_idxs = np.random.choice(self.size, batch_size, replace=False, p=p_i)
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
        )
        
        weights = torch.as_tensor(np.power(1. / p_i[sample_idxs] / self.size, self.beta), device=self.device)
        weights = weights / torch.max(weights)
        ############################
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha

        self.priorities[data_idxs] = priorities
        self.max_priority = np.max(self.priorities)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'


# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, action_size, device, seed):
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #
        super().__init__(capacity, eps, alpha, beta, state_size, device=device)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma
        ############################

    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        ############################
        # YOUR IMPLEMENTATION HERE #
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))
        ############################

    # def the other necessary class methods as your need
    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition, discard those rewards after done=True"""
        ############################
        # YOUR IMPLEMENTATION HERE #
        discount = 1
        terminated = False
        state, action, _, _ = self.n_step_buffer[0]
        reward = 0
        for sard in self.n_step_buffer:
            _, _, r, d = sard
            if not terminated:
                reward += r * discount
            discount *= self.gamma
            if d:
                terminated = True
        done = 1 if terminated else 0
        # self.n_step_buffer.clear()
        ############################
        return state, action, reward, done