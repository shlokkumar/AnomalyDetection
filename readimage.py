from os import listdir
from PIL import Image as PImage
import keras
import numpy as np
from os.path import isfile, join
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import image	
import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
import os

resize_h_n = 216 # 480
resize_w_n = 288 # 640

store_all_outputs = np.load("/home/mazda/Documents/Output/trainoutputimages.npy", allow_pickle = True)
store_all_losses = np.load("/home/mazda/Documents/Output/trainoutputmetrics.npy", allow_pickle = True)
image_names = np.load("/home/mazda/OneDrive/Ono Project/Anomaly Detection/Code Base/Shlok /ImageFilesNumpy/normalimagestrain.npy", allow_pickle = True)

print(store_all_outputs.shape)
print(store_all_losses.shape)
print(image_names.shape)
# print(store_all_losses)
print(type(store_all_losses))
print("Minimum Loss: " + str(np.min(store_all_losses)) + " at index: " + str(np.argmin(store_all_losses)))
print("Maximum Loss: " + str(np.max(store_all_losses)))


max_images = store_all_outputs.shape[0]	# to see the number of steps
pos = 0	# initialize the starting point of the image file name counter
max_pos = 640	# maximum length of the image file names list

# extracting last epoch output values
# output_image = store_all_outputs[-5:]

# extracting the last 5 loss values
# output_losses = store_all_losses[-5:]

# print(output_images.shape)
# print(output_losses.shape)

# for i in range(output_images.shape[0]):

for i in range(max_images):
  if (i+1) % 10 == 0 and i != 0:
    i_batch = store_all_outputs[i]
    for j in range(i_batch.shape[0]):
      if j == 30:
        file_save = r"/home/mazda/Documents/Natto Image Data/Normal Images-Resize Train Output-3/" + str(i) + "-" + str(store_all_losses[i]) + ":" + str(image_names[pos]) + ".jpg"
        data = i_batch[j]
        data = np.reshape(data, (resize_h_n, resize_w_n, 3))
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(file_save)
        pos = pos + 1
      print(str(i) + ":" + str(j) + ":" + str(pos))
    if pos == max_pos:
      pos = 0

