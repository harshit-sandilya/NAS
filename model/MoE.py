import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = torch.sum(gating_weights.unsqueeze(1) * expert_outputs, dim=2)
        return output, gating_weights


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 128
    output_dim = 32000
    num_experts = 8

    # Create the MoE model
    model = MixtureOfExperts(input_dim, output_dim, num_experts)

    # Example input
    x = torch.randn(5, input_dim)

    # Forward pass
    output, gating_weights = model(x)
    print("output ==>",output.shape, "gating_weights ==>",gating_weights.shape)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example target output
    y = torch.randn(5, output_dim)

    # Training loop with entropy regularization
    lambda_entropy = 0.1  # Regularization strength
    for epoch in range(100+1):
        optimizer.zero_grad()
        output, gating_weights = model(x)
        loss = criterion(output, y)

        # Entropy regularization
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8)) / x.size(0)
        total_loss = loss + lambda_entropy * entropy

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Entropy: {entropy.item()}')
