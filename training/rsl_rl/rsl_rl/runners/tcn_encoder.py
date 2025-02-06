import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal

class TcnEncoder(nn.Module):
    def __init__(
            self,
            history_length = 50,
            obs_dim = 50,
            device = 'cpu',
            z_dim = 12,
            lr = 1e-4,
    ):
        super().__init__()
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.device = device
        self.z_dim = z_dim

        self.conv1 = torch.nn.Conv1d(in_channels=50, out_channels=32, kernel_size=8, stride=4, padding=0,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)
        self.relu1 = nn.ReLU()

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)

        self.relu2 = nn.ReLU()

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)

        self.relu3 = nn.ReLU()

        self.linear1 = nn.Linear(96,32)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(32,self.z_dim)

        self.optim = Adam(self.parameters(), lr=lr)


    def forward(self, history):
        # if self.device is not None:
        #     history = torch.as_tensor(
        #         history,
        #         device=self.device,  # type: ignore
        #         dtype=torch.float32,
        #     )

        history = history.permute(0,2,1)
        # history = history.unsqueeze(3)

        x = self.conv1(history)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)

        x = x.flatten(1)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)


        return x