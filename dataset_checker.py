import sys, os

class dataset_checker():

  def check(self, path_rgb, path_depth):

    # Robustness checks
    if not os.path.isdir(path_rgb) or not os.path.isdir(path_depth):
      return -1

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
    
    return 0


  def fix(self, path_rgb, path_depth):

    # Robustness checks
    if not os.path.isdir(path_rgb) or not os.path.isdir(path_depth):
      return -1

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

    return 0
