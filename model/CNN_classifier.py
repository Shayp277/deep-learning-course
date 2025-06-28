import torch
import torch.nn as nn

class CNN_classifier(nn.Module):
    def __init__(self, input_size, output_size,dropout_rate):
        super().__init__()

        layers = []

        # First conv layer
        layers.append(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=13, stride=1, padding='same'))
        layers.append(nn.ReLU())
        if dropout_rate != 0.0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Second conv layer
        layers.append(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=1, padding='same'))
        layers.append(nn.ReLU())
        if dropout_rate != 0.0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Max Pooling 1D
        layers.append(GlobalMaxPool1d())
        layers.append(nn.Flatten())

        # Dense Layer
        layers.append(nn.Linear(16,16))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(16, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        return torch.max(x, dim=2)[0]