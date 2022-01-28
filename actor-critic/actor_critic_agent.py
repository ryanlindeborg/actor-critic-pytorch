import torch as th
import torch.nn as nn
import torch.nn.functional as F
from actor_critic_networks import ActorCriticNetworks

class ActorCriticAgent:
    def __init__(self, state_dims, n_actions, lr, gamma):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.actor_critic_networks = ActorCriticNetworks(lr=lr, state_dims=state_dims, n_actions=n_actions)
        # Gamma is discount factor of reward at each time step
        self.gamma = gamma
        self.log_action_probability = None
        self.current_state_value = None

    def select_action(self, state):
        state = th.tensor([state], dtype=th.float).to(self.device)
        action_probabilities, self.current_state_value = self.actor_critic_networks(state)
        action_probabilities = F.softmax(action_probabilities, dim=1)
        action_categorical_distribution = th.distributions.categorical.Categorical(probs=action_probabilities)
        action = action_categorical_distribution.sample()
        self.log_action_probability = action_categorical_distribution.log_prob(action)
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic_networks.optimizer.zero_grad()
        # If episode is done, value of future state is 0,
        # so only take future state value into account if episode is not done
        state_ = th.tensor([state_], dtype=th.float).to(self.device)
        _, next_step_value = self.actor_critic_networks(state_)
        reward = th.tensor([reward], dtype=th.float).to(self.device)
        bootstrapped_value = reward + (self.gamma * next_step_value * (1-int(done)))
        advantage = bootstrapped_value - self.current_state_value

        # TODO: Try different loss functions (MSE loss vs. smooth L1 loss)
        # Critic loss
        mse_loss = nn.MSELoss()
        critic_loss = mse_loss(self.current_state_value, bootstrapped_value)

        # Actor loss
        actor_loss = -(self.log_action_probability * advantage)

        # Do backprop on both critic and actor loss by adding losses together
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        self.actor_critic_networks.optimizer.step()