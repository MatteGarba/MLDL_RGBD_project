from torchvision.datasets import VisionDataset
import numpy as np
from random import randint
import torch
import torchvision

'''
The present class to bring together the rgb and depth images in a unique set (RGB_image, Depth_image) con label.
They can be o no rotated depending on the value of the flag "flag_rotate".
* if flag_rotate == true, images are rotated and label represent the groudtruth for the relative rotation
* Otherwise, they are not rotated and the label identifies the class of the object in the images (which is the same for  both the modalities)

'''
def Make_rotation(image):
   val = randint(0, 3)

   # Rotate it by val*90 degrees (0, 90, 180, 270)
   rotated = np.rot90(image, val)
   return np.array(rotated).transpose((2, 1, 0)), val

class DualDataset(VisionDataset):
    def __init__(self, rgb_dataset, depth_dataset, flag_rotate = False):
        super(DualDataset, self).__init__(rgb_dataset)
        if len(rgb_dataset)!=len(depth_dataset):
          raise ValueError("Differing lengths in datasets.")
        self.rgb = rgb_dataset
        self.depth = depth_dataset
        self.flag_rotate = flag_rotate

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

    def __len__(self):
        return len(self.rgb)
