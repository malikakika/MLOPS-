import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n_kernels, kernel_size=5),  
            nn.ReLU(),
            nn.MaxPool2d(2),                         
            nn.Conv2d(n_kernels, n_kernels, kernel_size=5),  
            nn.ReLU(),
            nn.MaxPool2d(2),                        
            nn.Flatten(),
            nn.Linear(n_kernels * 4 * 4, 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        return self.net(x)
