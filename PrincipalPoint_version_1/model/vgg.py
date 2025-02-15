from torch import nn
from torchvision import models


class InceptionV3(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""

    def __init__(self,):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(weights=models.VGG16_BN_Weights)
        self.fc = nn.Linear(1000, 2)
        self.act = nn.Sigmoid()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # 1,32*256*256

    def forward(self, x, isTrain):
        if isTrain:
            output = self.model(x)
            output = self.act(self.fc(output.logits))
        else:
            output = self.model(x)
            output = self.act(self.fc(output))
        return output
