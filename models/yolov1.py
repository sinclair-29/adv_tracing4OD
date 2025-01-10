import torch
import torch.nn as nn

class Yolov1(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Yolov1Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Conv 7x7x64-s-2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv 3x3x192
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Yolov1Backbone(nn.Module):
    NUM_BOXES = 2
    NUM_CLASS = 20
    output_size_per_cell = 5 * NUM_BOXES + NUM_CLASS

    def __init__(self):
        super().__init__()

        components = []
        components += [
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        for _ in range(4):
            components += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        components += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        for _ in range(2):
            components += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        components += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        ]
        for _ in range(2):
            components += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        self.conv_layers = nn.Sequential(*components)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, 7 * 7 * Yolov1Backbone.output_size_per_cell)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return torch.reshape(x, (x.size(dim=0), 7, 7, Yolov1Backbone.output_size_per_cell))
