import sys, os
from PIL import Image
import numpy as np

class Dataset_handler():

  def check(self, path_rgb, path_depth):

    # Robustness checks
    if not os.path.isdir(path_rgb) or not os.path.isdir(path_depth):
      sys.exit(-1)

    folders = os.listdir(path_rgb) # just one list, sice there are the same folders in both rgb and 
    nopng = 0 # counts how many non .png files are found
    flag = 0

    for folder in folders:
      cnt = 0 # counts how many correct couples there are (per folder)
      imgs_rgb = os.listdir(path_rgb + "/" + str(folder)) 
      imgs_depth = os.listdir(path_depth + "/" + str(folder))
      if len(imgs_rgb) != len(imgs_depth):
        print(f"An error will occur for class: {str(folder)}")
        print(f"#RGB images: {len(imgs_rgb)} ; #DEPTH images: {len(imgs_depth)}")
      for img_rgb, img_depth in zip(imgs_rgb, imgs_depth):
        cnt = cnt + 1

        # check if there are files not ".png"
        tail = (img_rgb.split("."))[1]
        if tail != "png":
          nopng = nopng + 1
          flag = 1
          tail = (img_depth.split("."))[1]
        if tail != "png":
          nopng = nopng + 1
          flag = 1
        
        if str(img_rgb) != str(img_depth):
          print(f"Discrepancy at image {cnt} of class {str(folder)}")
          print(str(img_rgb))
          print()
          flag = 1
          break
          # sys.exit(1)

    if nopng > 0:
      print(f"WARNING: there are {nopng} non '.png' files")
      print()

    if flag == 0:
      print("\nCheck complete --> everything is ok")
    else:
      print("\nCheck complete --> errors found")
    
    return


  def fix(self, path_rgb, path_depth):

    # Robustness checks
    if not os.path.isdir(path_rgb) or not os.path.isdir(path_depth):
      sys.exit(-1)

    folders = os.listdir(path_rgb) # just one list, sice there are the same folders in both rgb and 

    cnt_corrected = 0

    for folder in folders:
      occurrences = dict()
      imgs_rgb = os.listdir(path_rgb + "/" + str(folder)) 
      imgs_depth = os.listdir(path_depth + "/" + str(folder))
      
      # counting the occurrences
      for img in imgs_rgb:
        if str(img) in occurrences:
          occurrences[str(img)] = occurrences[str(img)] + 1
        else:
          occurrences[str(img)] = 1

      for img in imgs_depth:
        if str(img) in occurrences:
          occurrences[str(img)] = occurrences[str(img)] + 1
        else:
          occurrences[str(img)] = 1

      # correcting
      for img in imgs_rgb:
        if str(img) not in occurrences or occurrences[str(img)] < 2:
          os.remove(path_rgb+"/"+str(folder)+"/"+str(img))
          cnt_corrected = cnt_corrected + 1

      for img in imgs_depth:
        if str(img) not in occurrences or occurrences[str(img)] < 2:
          os.remove(path_depth+"/"+str(folder)+"/"+str(img))
          cnt_corrected = cnt_corrected + 1
    print(f"Removed {cnt_corrected} images")

    return
  
  
  def reshape(self):
    
    import os, shutil

    # Reshaping synROD:

    # Create the folders for the new synRODs
    if os.path.isdir("dataset_rgb_synROD") == False:
            os.makedirs("dataset_rgb_synROD");
    if os.path.isdir("dataset_depth_synROD") == False:
            os.makedirs("dataset_depth_synROD");

    # explore the unzipped synROD and copy the images in a correct structure
    base_path = "/content/dataset_unzipped/synROD/"
    folders = [dI for dI in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,dI))] # list of all the folders (classes) of synROD
    for folder in folders:                                                                      # for each class/folder:
      if folder == ".git":
        continue
      if os.path.isdir("dataset_rgb_synROD/"+str(folder)) == False:                             # create the two correponding in rgb and depth datasets
            os.makedirs("dataset_rgb_synROD/"+str(folder));
      if os.path.isdir("dataset_depth_synROD/"+str(folder)) == False:
            os.makedirs("dataset_depth_synROD/"+str(folder));

      images = os.listdir(base_path+str(folder)+"/rgb")                                         # list of all the rgb images
      for img in images:
        shutil.copy(base_path+str(folder)+"/rgb/"+str(img), "dataset_rgb_synROD/"+str(folder))  # copy each image in "dataset_rgb_synROD/_class_/"

      images = os.listdir(base_path+str(folder)+"/depth")                                       # list of all the depth images
      for img in images:
        shutil.copy(base_path+str(folder)+"/depth/"+str(img), "dataset_depth_synROD/"+str(folder))# copy each image in "dataset_depth_synROD/_class_/"


    # Reshaping ROD:

    # Create the folders for the new synRODs
    if os.path.isdir("dataset_rgb_ROD") == False:
            os.makedirs("dataset_rgb_ROD");
    if os.path.isdir("dataset_depth_ROD") == False:
            os.makedirs("dataset_depth_ROD");

    # explore the unzipped rgb ROD and copy the images in a correct structure
    base_path = "/content/dataset_unzipped/ROD/ROD_rgb/"
    folders = [dI for dI in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,dI))]
    for folder in folders:                                                                      # for each class/folder:
      if folder == ".git":
        continue
      if os.path.isdir("dataset_rgb_ROD/"+str(folder)) == False:                                # create the corresponding folder in rgb rod dataset
            os.makedirs("dataset_rgb_ROD/"+str(folder));

      subfolders = os.listdir(base_path+str(folder))                                            # list all the subfolders for each class
      for subfolder in subfolders:                                                              # and for each one of them:
        images = os.listdir(base_path+str(folder)+"/"+str(subfolder))
        for img in images:

          pieces = str(img).split("_")
          i = 0
          new_name = ""
          while i < len(pieces)-1:
            new_name = new_name + pieces[i] + "_"
            i = i + 1
          new_name = new_name + ".png"
          shutil.copy(base_path+str(folder)+"/"+str(subfolder)+"/"+str(img), "dataset_rgb_ROD/"+str(folder))# copy all its images into "dataset_rgb_ROD/_class_/"
          os.rename("/content/dataset_rgb_ROD/"+str(folder)+"/"+str(img), "/content/dataset_rgb_ROD/"+str(folder)+"/"+str(new_name))                     # rename the image (removing the suffix)

    # explore the unzipped depth ROD and copy the images in a correct structure
    base_path = "/content/dataset_unzipped/ROD/ROD_surfnorm/"
    folders = [dI for dI in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,dI))]
    for folder in folders:                                                                      # for each class/folder:
      if folder == ".git":
        continue
      if os.path.isdir("dataset_depth_ROD/"+str(folder)) == False:                              # create the corresponding folder in depth rod dataset
            os.makedirs("dataset_depth_ROD/"+str(folder));

      subfolders = os.listdir(base_path+str(folder))                                            # list all the subfolders for each class
      for subfolder in subfolders:                                                              # and for each one of them:
        images = os.listdir(base_path+str(folder)+"/"+str(subfolder))
        for img in images:

          pieces = str(img).split("_")
          i = 0
          new_name = ""
          while i < len(pieces)-1:
            new_name = new_name + pieces[i] + "_"
            i = i + 1
          new_name = new_name + ".png"
          shutil.copy(base_path+str(folder)+"/"+str(subfolder)+"/"+str(img), "dataset_depth_ROD/"+str(folder))# copy all its images into "dataset_depth_ROD/_class_/"
          os.rename("/content/dataset_depth_ROD/"+str(folder)+"/"+str(img), "/content/dataset_depth_ROD/"+str(folder)+"/"+str(new_name))                     # rename the image (removing the suffix)
    
    return
  
