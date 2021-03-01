import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

class ImgEncoder(nn.Module):
    def __init__(self, G, img_enc_dim):
        super(ImgEncoder, self).__init__()

        assert G in [4, 2, 1]
        last_stride = 1 if G == 4 else 2

        #self.last = nn.Conv2d(256, img_enc_dim, 3, last_stride, 1)
        if G in [4, 1]:
            self.last = nn.Sequential(
                nn.Conv2d(512, 512, 3, last_stride, 1),
                nn.Conv2d(512, img_enc_dim, 3, last_stride, 1),
            )
        else:
            self.last = nn.Sequential(
                nn.Conv2d(512, 512, 3, last_stride, 1),
                nn.Conv2d(512, img_enc_dim, 3, 1, 1),
            )

        resnet = resnet18()
        self.enc = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x):
        """
        Get image feature
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, img_enc_dim, G, G)
        """
        B = x.size(0)
        x = self.enc(x)
        x = self.last(x)
        return x
