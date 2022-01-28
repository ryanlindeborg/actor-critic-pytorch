import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Actor critic class creates one network for Q value function, and a separate network for (state) value function
class ActorCriticNetworks(nn.Module):
    def __init__(self, state_dims, n_actions, lr):
        super(ActorCriticNetworks, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.observation_dims = state_dims
        # Number of actions for actor network
        self.n_actions = n_actions
        self.lr = lr

        hidden_layer_1_size = 128
        hidden_layer_2_size = 128
        self.fc1_layer = nn.Linear(in_features=self.observation_dims[0], out_features=hidden_layer_1_size)
        self.fc2_layer = nn.Linear(in_features=hidden_layer_1_size, out_features=hidden_layer_2_size)
        self.output_actor_layer = nn.Linear(in_features=hidden_layer_2_size, out_features=self.n_actions)
        self.output_critic_layer = nn.Linear(in_features=hidden_layer_2_size, out_features=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input_tensor):
        # Here we share the same base layers between the actor and critic network
        input_tensor = input_tensor.to(self.device)
        output_tensor = F.relu(self.fc1_layer(input_tensor))
        output_tensor = F.relu(self.fc2_layer(output_tensor))
        action_probabilities = self.output_actor_layer(output_tensor)
        state_value = self.output_critic_layer(output_tensor)

        return action_probabilities, state_value



