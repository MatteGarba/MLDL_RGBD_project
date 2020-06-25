from torchvision.datasets import VisionDataset
import numpy as np
from random import randint
import torch
import torchvision
import random
import numbers
import torchvision.transforms.functional as FF
from PIL import Image

'''
The present class to bring together the rgb and depth images in a unique set (RGB_image, Depth_image) con label.
They can be o no rotated depending on the value of the flag "flag_rotate".
* if flag_rotate == true, images are rotated and label represent the groudtruth for the relative rotation
* Otherwise, they are not rotated and the label identifies the class of the object in the images (which is the same for  both the modalities)

'''
def permute(img, code):
  
  # All the permutations (just for reference)
  # permutations = {1:'RGB', 2:'GRB', 3:'BRG', 4:'RBG', 5:'GBR', 6:'BGR'}

  r, g, b = Image.Image.split(img)

  if code == 1:
    img = Image.merge("RGB", (r,g,b))   # 1:'RGB'
  elif code == 2:
    img = Image.merge("RGB", (g,r,b))   # 2:'GRB'
  elif code == 3:
    img = Image.merge("RGB", (b,r,g))   # 3:'BRG'
  elif code == 4:
    img = Image.merge("RGB", (r,b,g))   # 4:'RBG'
  elif code == 5:
    img = Image.merge("RGB", (g,b,r))   # 5:'GBR'
  elif code == 6:
    img = Image.merge("RGB", (b,g,r))   # 6:'BGR'
  else:
    print("WARNING: this must not happen!!!")
    img = img
  
  return img

def make_permutation(rgb_im, depth_im, second_variation=False):
  # This method can be used in two ways:
  # 1. --> if the flag is False: the two images have a 50% chance to undergo
  #        the same transformation. 
  #        The label is 0 if it's the same and 1 otherwise and the task will
  #        have to uderstand if they underwent the same permutation.
  # 2. --> if the flag is True: the two images undergo the same transformation.
  #        The pretext task must understand which of the 6 possible permutations
  #        was applied

  if second_variation == False:
    # we want the image to ha 50% chance to undergo the same transformation
    # otherwise the two classes would be unbalanced
    discriminator = randint(1,10)
    if discriminator % 2 == 1:
      # if odd, they will undergo the same permutation
      choice = randint(1, 6)
      choice_rgb = choice
      choice_depth = choice
    else:
      # if even, they will undergo different permutations
      choice_rgb = randint(1,6)
      choice_depth = randint(1,6)
      while choice_rgb == choice_depth:
        choice_depth = randint(1,6)

    rgb_im = permute(rgb_im, choice_rgb)
    depth_im = permute(depth_im, choice_depth)

    # label assignment
    label = 0                           # different permutation undergone
    if choice_rgb == choice_depth:      
      label = 1                         # same permutation undergone

  else:
    choice = randint(1, 6)
    rgb_im = permute(rgb_im, choice)
    depth_im = permute(depth_im, choice)
    label = choice-1                    # labels will go from 0 to 5

  return rgb_im, depth_im, label


def Make_rotation(image):
   val = randint(0, 3)

   # Rotate it by val*90 degrees (0, 90, 180, 270)
   rotated = np.rot90(image, val)
   return np.array(rotated).transpose((2, 1, 0)), val

class DualDataset(VisionDataset):
    def __init__(self, rgb_dataset, depth_dataset, flag_rotate=False, dual_transforms=None, flag_permute=False, variation2=False):
      """
      dual_transforms: object of DualCompose class.
      """
      super(DualDataset, self).__init__(rgb_dataset)
      if len(rgb_dataset)!=len(depth_dataset):
        raise ValueError("Differing lengths in datasets.")
      self.rgb = rgb_dataset
      self.depth = depth_dataset
      self.flag_rotate = flag_rotate
      self.transforms = dual_transforms
      self.flag_permute = flag_permute
      self.variation2 = variation2

    def __getitem__(self, key):
        # Combine (rgb_image, label) and (depth_image, label).
        # @Returns: ( (rgb_image, depth_image), label )
        rgb_tuple = self.rgb[key]
        depth_tuple = self.depth[key]

        if rgb_tuple[1]!=depth_tuple[1]:
            raise ValueError("Differing labels for pictures with same key.")

        rgb_image = rgb_tuple[0]
        depth_image = depth_tuple[0]
        label = rgb_tuple[1]

        if self.flag_permute == True:
          rgb_image, depth_image, label = make_permutation(rgb_image, depth_image, self.variation2)

        if self.transforms is not None:
            rgb_image, depth_image = self.transforms(rgb_image, depth_image)

        if self.flag_rotate == True:
          #rotate rgb image
          img = torchvision.utils.make_grid(rgb_image)
          img = img.numpy().transpose((1, 2, 0))
          rotated, label1 = Make_rotation(img)
          rgb_image = torch.Tensor(rotated)
          # rotate depth image
          img = torchvision.utils.make_grid(depth_image)
          img = img.numpy().transpose((1, 2, 0))
          rotated, label2 = Make_rotation(img)
          depth_image = torch.Tensor(rotated)
          #label
          label = (label1 - label2) % 4

        return ((rgb_image, depth_image), label)
      
      
    def sample_images(self, n_samples=1):
      """
        This function samples the given dataset in parallel on the two modalities. It is used in conjunction with t-SNE. Due to COLAB limitations and the nature of the algorithm,
        requiring high values of n_samples is not feasible. 
      """
      # Consistency check
      if len(self) < n_samples:
        raise ValueError("Cannot provide this many samples.")
      if n_samples > 250:
        raise RuntimeWarning("This value of samples is high and may cause CUDA out memory errors.")
        
      # Find the indexes of the required images
      indexes = set()
      for i in range(n_samples):
        j = randint(0, len(self)-1)
        while j in indexes:
          j = randint(0, len(self)-1)
        indexes.add(j)

      # Build the lists to be returned
      rgb_samples = []
      depth_samples = []
      for i in indexes:
        (rgb_tensor, d_tensor), _ = self[i]
        rgb_samples.append(rgb_tensor)
        depth_samples.append(d_tensor)

      return rgb_samples, depth_samples 

    
    def __len__(self):
        return len(self.rgb)


class DualRandomHFlip(object):
    """This class applies the same random horizontal flip to both modalities of RGB-D.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, rgb, depth):
        if random.random() < self.p:
            return FF.hflip(rgb), FF.hflip(depth)
        return rgb, depth
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def _get_image_size(img):
      if FF._is_pil_image(img):
          return img.size
      elif isinstance(img, torch.Tensor) and img.dim() > 2:
          return img.shape[-2:][::-1]
      else:
          raise TypeError("Unexpected type {}".format(type(img)))


class DualRandomCrop(object):
    """
    This class applies the same random crop transformation to both modalities of RGB-D.
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb, depth):
        """
        rgb, depth (PIL Image): images to be cropped in the same way. The must have the same dimension!
        """
        if (rgb.size[0] != depth.size[0]) or (rgb.size[1] != depth.size[1]):
            raise ValueError("Images have different sizes.")

        if self.padding is not None:
            rgb = FF.pad(rgb, self.padding, self.fill, self.padding_mode)
            depth = FF.pad(depth, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and rgb.size[0] < self.size[1]:
            rgb = FF.pad(rgb, (self.size[1] - rgb.size[0], 0), self.fill, self.padding_mode)
            depth = FF.pad(depth, (self.size[1] - depth.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and rgb.size[1] < self.size[0]:
            rgb = FF.pad(rgb, (0, self.size[0] - rgb.size[1]), self.fill, self.padding_mode)
            depth = FF.pad(depth, (0, self.size[0] - depth.size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(rgb, self.size)
        return FF.crop(rgb, i, j, h, w), FF.crop(depth, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def dual(obj, is_dual=False):
  """
  This function adds an attribute to transforms, to be used by DualCompose.
  """
  obj.is_dual = is_dual
  return obj

class DualCompose(object):
  """
  This class is a wrapper for transforms to be applied on dual modality images.
  It applies the same transforms on both modality if is_dual==True.
  It considers them individually otherwise.
  """
  def __init__(self, transforms):
      self.transforms = transforms

  def __call__(self, rgb, depth):
      for t in self.transforms:
          if t.is_dual==False:
              rgb = t(rgb)
              depth = t(depth)
          else:
              rgb, depth = t(rgb, depth)
      return rgb, depth

  def __repr__(self):
      format_string = self.__class__.__name__ + '('
      for t in self.transforms:
          format_string += '\n'
          format_string += '    {0}'.format(t)
      format_string += '\n)'
      return format_string
