import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision import models
from torchvision import datasets

from PIL import Image
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np

from random import randint

from sklearn.manifold import TSNE
DEVICE = 'cuda'


class FeatureVisualization():
  """
  Examples of usage:
    (Saliency Map)
    >>>   visualizer = FeatureVisualization()
    >>>   visualizer.saliency_maps(rgb_batch[0], depth_batch[0], Extractor, mainTask)

    (t-SNE)
    >>>   visualizer = FeatureVisualization()
    >>>   stratific = source_dataloarder.dataset.get_stratification()
    >>>   rgb_samples, depth_samples = test_target_dataloarder.dataset.sample_images(stratific, n_samples=250)
    >>>   plottable_target = visualizer.features_2d(rgb_samples, depth_samples, Extractor, mainTask)

  """
  def __init__(self):
    self.rgb_grads = None
    self.d_grads = None

  def rgb_hooker(self, grad):
    self.rgb_grads = grad

  def depth_hooker(self, grad):
    self.d_grads = grad

  def deprocess(self, image):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage(),
    ])
    return transform(image)

  def show_images(self, rgb, depth, saliency_rgb, saliency_depth):
    fig, ax = plt.subplots(2,2, figsize=(18,15))
    ax[0,0].imshow(self.deprocess(rgb.detach().cpu()))
    ax[0,1].imshow(self.deprocess(depth.detach().cpu()))
    ax[1,0].imshow((saliency_rgb.cpu()))
    ax[1,1].imshow(saliency_depth.cpu())

  def saliency_maps(self, rgb, depth, feat_extractor, classifier):
    """
      rgb, depth: [3,224,224] tensors taken from our custom dataloader, so transformations have already been applied.
      feat_extractor: the part of the network that extracts the features to be fed to the heads (classifiers)
      classifier: main head of the network, which does not include terminal normalizations like softmax or sigmoid. It is used to obtain the scores for each class.
    """
    feat_extractor = feat_extractor.to(DEVICE)
    classifier = classifier.to(DEVICE)
    feat_extractor.eval()
    classifier.eval()

    rgb = rgb.unsqueeze(0)
    rgb.requires_grad_()
    rgb.retain_grad()
    rgb.register_hook(self.rgb_hooker)

    depth = depth.unsqueeze(0)
    depth.requires_grad_()
    depth.retain_grad()
    depth.register_hook(self.depth_hooker)

    features = feat_extractor.forward(rgb, depth)
    scores = classifier.forward(features)
    value, _ = torch.max(scores, axis=1)
    value.backward()

    saliency_rgb, _ = torch.max(rgb.grad.data.abs(),dim=1)      # (1,1,224,224)
    saliency_depth, _ = torch.max(depth.grad.data.abs(),dim=1)  # (1,1,224,224)

    self.show_images(rgb, depth, saliency_rgb[0], saliency_depth[0])

  def features_2d(self, rgbs, depths, feat_extractor, classifier):
    feat_extractor.eval()
    classifier.eval()

    feat_arr = []
    for rgb, d in zip(rgbs, depths):
      rgb = rgb.to(DEVICE)
      d = d.to(DEVICE)
      #print(rgb.shape)
      features = feat_extractor(rgb.unsqueeze(0), d.unsqueeze(0))
      for i in range(4):
        features = classifier.layer[i](features)
      feat_arr.append(features)

    feat_arr = np.array([entry[0].detach().cpu().numpy() for entry in feat_arr])
    tsne = TSNE(n_components=2)
    plottable = tsne.fit_transform(feat_arr)

    """fig, ax = plt.subplots(figsize=(18,15))
    ax.scatter(plottable[:,0], plottable[:,1])
    plt.axis('off')
    plt.show()"""
    return plottable
