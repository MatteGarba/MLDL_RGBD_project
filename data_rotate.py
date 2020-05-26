import sys, os
from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np
from random import randint



##Another class which gives us and object of type Dataset containing rotated images
  
def Make_rotation(image):
   val = randint(0, 3)

   # Rotate it by val*90 degrees (0, 90, 180, 270)
   rotated = np.rot90(image, val)
   return np.array(rotated).transpose((2, 1, 0)), val

  
class Rotate(VisionDataset):
    def __init__(self, dataset,  transform=None):
        super(Rotate, self).__init__(dataset)
        self.dataset = dataset
        self.wrapper = []

        for i in range(len(data_synRODrgb)):
          image = data_synRODrgb[i][0]
          img = torchvision.utils.make_grid(image)
          #print(img.shape)
          img = img.numpy().transpose((1, 2, 0))
          #print(img.shape)
          rotated, label = Make_rotation(img)
          self.wrapper.append((torch.Tensor(rotated),label))
          
    
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image, label = self.wrapper[index]
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image but actually is a tensor
                           # label can be int
        image = transforms.ToPILImage()(image).convert("RGB")
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label   
      
    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.wrapper) # Provide a way to get the length (number of elements) of the dataset
        return length
