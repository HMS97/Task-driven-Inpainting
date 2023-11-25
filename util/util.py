"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from . import augmentations
from torchvision import transforms
from time import sleep
from IPython import display



mixture_width = 2
mixture_depth = -1
aug_severity = 10

def aug_plus(image, preprocess,mode = 'n'):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """

  # if type(image) == type(np.ndarray):
  #   image = Image.fromarray(image)
  # else:
  #   image = Image.fromarray(image)
  aug_list = augmentations.augmentations
  # if args.all_ops:
    # aug_list = augmentations.augmentations_all
  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))
  mix = np.zeros((np.array(image).shape[0],np.array(image).shape[1]))
  mixed = np.zeros((np.array(image).shape[0],np.array(image).shape[1]))
  mixture_list = []
  weights = [0.2,0.1,0.3,0.4]
  for i in range(mixture_width):
    mixture_list.append(np.zeros((np.array(image).shape[0],np.array(image).shape[1])))
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)

    for _ in range(3):
      op = np.random.choice(aug_list)

      mixture_list[i] +=  (np.array(op(image_aug,aug_severity))> 0 ).astype(int)

  for i in range(mixture_width):
    mixed += weights[i]*mixture_list[i]

  mixed = mixed + np.array(image_aug)*weights[-1] 
  mixed = mixed > 0.5
  mixed =  preprocess(image_aug)
  return mixed

def aug(image, preprocess,mode = 'n'):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  if type(image) == type(np.ndarray):
    image = Image.fromarray(image)
  else:
    image = Image.fromarray(image)
  aug_list = augmentations.augmentations
  # if args.all_ops:
    # aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))
  mix = torch.zeros_like(preprocess(image))
  mixed = torch.zeros_like(preprocess(image))

  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity, image.size[0])
    # Preprocessing commutes since all coefficients are convex
      # print(op,preprocess(image_aug).shape, ws[i])




    # if mode == 'v':
    #   mix = preprocess(image_aug)
    #   mixed[:,int(i*image.size[1]/mixture_width):int((i+1)*image.size[1]/mixture_width),:] =  mix[:,int(i*image.size[1]/mixture_width):int((i+1)*image.size[1]/mixture_width),:] 
    # if mode == 'h':
    #   mix = preprocess(image_aug)
    #   mixed[:,:,int(i*image.size[0]/mixture_width):int((i+1)*image.size[0]/mixture_width)] =  mix[:,:,int(i*image.size[0]/mixture_width):int((i+1)*image.size[0]/mixture_width)] 
    if mode == 'n':
      mix +=  preprocess(image_aug)
      mixed =  mix


  return mixed

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
