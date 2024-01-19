import torch
import torch.nn as nn
import torch.optim as optim

class StressTestModel(nn.Module):
    def __init__(self):
        super(StressTestModel, self).__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(1000, 900),
            nn.Linear(900, 800),
            nn.Linear(800, 700),
            nn.Linear(700, 600),
            nn.Linear(600, 500),
            nn.Linear(500, 400),
            nn.Linear(400, 300),
            nn.Linear(300, 200),
            nn.Linear(200, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 25),
            nn.Linear(25, 12),
            nn.Linear(12, 6),
            nn.Linear(6, 3),
            nn.Linear(3, 1)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        return x

def run_gpu_stress_test(model, device, num_iterations=10000000000):
    # Move model to the specified device
    model.to(device)

    # Dummy input tensor (replace with your actual input size)
    input_tensor = torch.randn(50000, 1000).to(device)

    # Dummy target tensor (replace with your actual target size)
    target_tensor = torch.randn(50000, 1).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Run stress test
    for iteration in range(num_iterations):
        # Forward pass
        output = model(input_tensor)

        # Compute loss
        loss = criterion(output, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

        # Clear cache
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Clear cache
    torch.cuda.empty_cache()

    # Instantiate the stress test model
    stress_test_model = StressTestModel()

    # Specify the GPU device (e.g., 'cuda:0')
    gpu_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Run stress test on the GPU
    run_gpu_stress_test(stress_test_model, gpu_device)
