import torch
import torch.nn as nn
from torchvision.models import resnet18

"""
The present python script contains the following classes:

* The FeatureExtractor class used to define the feature extractor (Resnet18 without the FC layer),***(Edoardo hai detto che lo facevi quindi te lo lascio)
* The MainTask class, used to perform the main task which a multi class classification
* The PreText class which is used to predict the relative rotation between the depth and the RGB modalities

"""

class PreText(nn.Module):
  """
  Pretext task
  """
  def __init__(self, num_classes = 4, featureMaps = 512, **kwargs):
    super(PreText, self).__init__()
    self.layer = nn.Sequential(
          nn.Conv2d(featureMaps*2, 100, kernel_size = 1, stride = 1),
          nn.BatchNorm2d(100),
          nn.ReLU(inplace=True),
          nn.Conv2d(100, 100, kernel_size = 3, stride = 2),
          nn.BatchNorm2d(100),
          nn.ReLU(inplace=True),
          nn.Flatten(),
	  nn.Dropout(),
          nn.Linear(100*3*3, 100),
          nn.BatchNorm1d(100),
          nn.ReLU(inplace=True),
	  nn.Dropout(),
          nn.Linear(100, num_classes),
      )
  def forward(self, h):
    c = self.layer(h)
    return c


class  MainTask(nn.Module):
  """
  Main classifier
  """
  def __init__(self, num_classes = 47, featureMaps = 512, **kwargs):
    super(MainTask, self).__init__()
    self.layer = nn.Sequential(
          nn.AdaptiveAvgPool2d((7,7)),
          nn.Flatten(),
	  nn.Dropout(),
          nn.Linear(featureMaps*2*7*7, 1000),
          nn.BatchNorm1d(1000),
          nn.ReLU(inplace=True),
	  nn.Dropout(),
          nn.Linear(1000, num_classes),
        )

  def forward(self, h):
     d = self.layer(h)
     return d



class Branch(nn.Module):
    """
    This class is for the branches that make the FeatureExtractor.
    Source code for resnet18: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(self, pretrained=True):
        super(Branch, self).__init__()
        net = resnet18(pretrained=pretrained, progress=True)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.conv2 = net.layer1
        self.conv3 = net.layer2
        self.conv4 = net.layer3
        self.conv5 = net.layer4


    def forward(self, x):
        """
        x: input data. 4-dimensional tensor ( shape: 64x3x224x224   i.e. batch_size,num_channels,widht,height )
        @Returns: 4-dimensional tensor of size [len(x), 512, 7, 7]
        """
        # the residual part is implemented in the BasicBlock class that composes layers layer1..4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        """
        conv5 is made of 2 conv. layers that apply 512 filters each, and give 7x7 outputs for each filter and for each image
        """
        return x


class FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.rgb_branch = Branch(pretrained=pretrained)
        self.depth_branch = Branch(pretrained=pretrained)

    def _combine_features(self, rgb_out, depth_out):
        """
        Note that len(rgb_out)==len(depth_out).
        @Returns: 4-dimensional tensor of size [len(rgb_out), 1024, 7, 7]   (is this what we mean with "combine along the channel dimension"?)
        """
        return torch.cat([rgb_out, depth_out], dim=1)

    def forward(self, rgb_batch, depth_batch):
        # Forward pass in both branches
        rgb_features = self.rgb_branch(rgb_batch)
        depth_features = self.depth_branch(depth_batch)

        # Combine the outputs of the two branches to make the final features.
        return self._combine_features(rgb_features, depth_features)
