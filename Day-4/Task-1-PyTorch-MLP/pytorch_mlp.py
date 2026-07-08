import torch
import torch.nn as nn

print("PyTorch Version:", torch.__version__)

# Sample input
x = torch.randn(5, 10)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleMLP()

output = model(x)

print("\nInput Shape:", x.shape)
print("Output Shape:", output.shape)
print("\nOutput Tensor:")
print(output)

print("\nMLP Model Created Successfully!")
