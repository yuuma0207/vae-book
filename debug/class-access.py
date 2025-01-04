import torch
from torch import nn

class DebugModule(nn.Module):
    def __init__(self):
        print(">>> __init__ invoked")
        super().__init__()
        
    def __call__(self, *args, **kwargs):
        print(">>> __call__ invoked")
        return super().__call__(*args, **kwargs)
    
    def forward(self, x):
        print(">>> forward invoked")
        return x


class ConvBlock(DebugModule):
    def __init__(self, in_ch, out_ch):
        print("a")
        super().__init__()
        print("b")
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        print(f">>> ConvBlock forward invoked with input shape {x.shape}")
        return self.convs(x)
    
class Unet(DebugModule):
    def __init__(self, in_ch=1):
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        print(f">>> Unet forward invoked with input shape {x.shape}")
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x = self.bot1(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.out(x)
        return x


model = Unet()
x = torch.randn(10, 1, 28, 28)
y = model(x)
print(y.shape)

