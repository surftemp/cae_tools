import torch
from torch import nn

# Ensure this class is defined before using it
class MeanPredictorSingleChannel(nn.Module):
    def __init__(self, fc_size, dropout_rate=0.5):
        super(MeanPredictorSingleChannel, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, fc_size),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, 1)  # Output a single value representing the mean
        )

    def forward(self, x):
        # Use only the first channel
        x = x[:, :1, :, :]  # Assuming input shape is (batch_size, channels, height, width)
        x = self.encoder_cnn(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x

# The rest of your UNET class and associated methods follow here...
