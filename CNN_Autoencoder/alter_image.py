from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.image as mpimg
import os
import tensorflow.compat.v1 as tf
import math
from PIL import Image
from os import listdir
from os.path import isfile, join
import keras
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import shutil
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing import image_dataset_from_directory
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model
from PIL import Image

# directory location to read images from
anomaly_dir = "your path" + '/Anomalous Images/'
normal_dir = "your path" + '/Normal Images/'

# func to crop the center part of fixed width and height
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
# calls func at line 35
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

# read input data (images)
file_name_normal = np.array([f for f in listdir(normal_dir) if isfile(join(normal_dir, f))])
file_name_anomalous = np.array([f for f in listdir(anomaly_dir) if isfile(join(anomaly_dir, f))])
np.random.shuffle(file_name_normal)
np.random.shuffle(file_name_anomalous)
file_name_normal = file_name_normal[:640]  # out of 870 images, we use only 640 images to train our network
file_name_anomalous = file_name_anomalous[:640]  # out of 870 images, we use only 640 images to train our network

# resizing to make an image of size (224, 224, 3)
resize_w = 224
resize_h = 224

## reading and resizing normal images
# saving the images as matrices

normal_images = []
for i in range(len(file_name_normal)):
  img = load_img(normal_dir + file_name_normal[i])
  im_new = crop_max_square(img)
  im_new = im_new.resize(size=(resize_w, resize_h))
  image = img_to_array(im_new)
  normal_images.append(image)

anomalous_images = []
for i in range(len(file_name_anomalous)):
  img = load_img(anomaly_dir + file_name_anomalous[i])
  im_new = crop_max_square(img)
  im_new = im_new.resize(size=(resize_w, resize_h))
  image = img_to_array(im_new)
  anomalous_images.append(image)

normal_images = np.array(normal_images)
anomalous_images = np.array(anomalous_images)

# # extracting dimensions
m = normal_images.shape[0]
h = normal_images[0].shape[0]
w = normal_images[0].shape[1]
c = normal_images[0].shape[2]

print(str(m) + ":" + str(h)+ ":" + str(w) + ":" + str(c))
print(normal_images.shape)  # must print (640, 224, 224, 3) if value at line 53 is unchanged

np.save('path_to_save_normal_images_tensor_to_be_read_later', normal_images ,allow_pickle=True)
np.save('path_to_save_anomalous_images_tensor_to_be_read_later', anomalous_images ,allow_pickle=True)
