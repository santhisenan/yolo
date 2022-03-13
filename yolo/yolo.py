from torch import nn


class YOLOBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, num_repeats):
        super().__init__()
        self.first_layer = nn.ReLU(
            nn.Conv2d(in_channels, out_channels_1, kernel_size=1)
        )
        self.second_layer = nn.ReLU(
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size=1)
        )
        self.unit = nn.Sequential(self.first_layer, self.second_layer)
        self.num_repeats = num_repeats

    def forward(self, input):
        X = input
        for _ in range(self.num_repeats):
            X = self.unit(X)
        return X


def get_yolov1(in_channels=3, S=7, C=20, B=2):

    # Conv block 1 in: [B, 3, 448, 448], out: [B, 64, 112, 112]
    block1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Conv block 2 in: [B, 64, 112, 112], out: [B, 192, 56, 56]
    block2 = nn.Sequential(
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Conv block 3 in: [B, 192, 56, 56], out: [B, 512, 28, 28]
    block3 = nn.Sequential(
        nn.Conv2d(192, 128, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Conv block 4 in: [B, 512, 28, 28], out: [B, 1024, 14, 14]
    block4 = nn.Sequential(
        YOLOBlock(512, 256, 512, 4),
        nn.Conv2d(512, 512, 1),
        nn.ReLU(),
        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Conv block 5 in: [B, 1024, 14, 14], out: [B, 1024, 7, 7]
    block5 = nn.Sequential(
        YOLOBlock(1024, 512, 1024, 2),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
        nn.ReLU(),
    )

    # Conv block 6 in: [B, 1024, 7, 7], out: [B, 1024, 7, 7]
    block6 = nn.Sequential(
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
    )

    model = nn.Sequential(
        block1,
        block2,
        block3,
        block4,
        block5,
        block6,
        nn.Flatten(),
        nn.Linear(1024 * 7 * 7, 4096),
        nn.LeakyReLU(),
        nn.Linear(4096, S * S * (B * 5 + C)),
        nn.Sigmoid(),
    )
    return model
