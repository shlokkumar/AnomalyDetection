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
from keras.applications.vgg16 import VGG16

normal_images = np.load('path_where_normal_images_tensor_is_saved', allow_pickle=True)
anomalous_images = np.load('path_where_anomalous_images_tensor_is_saved', allow_pickle=True)

# forming testing and training tensors
normal_images_train = normal_images[:640]
normal_images_test = normal_images[640:728]
anomalous_images_test = normal_images[640:728]

## defining the cnn network
input_shape = normal_images.shape[1:]
batch_size_model = 32
epoch_model = 100

# designing the encoder network
encoder_input = Input(shape=(input_shape), name='Input_Layer')
X = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="valid", activation='relu', name='conv_layer1')(input1)
X = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="valid", activation='relu', name='conv_layer2')(X)
X = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="valid", activation='relu',name='conv_layer3')(X)
X = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='valid',activation='relu', name = 'conv_layer4')(X)

#designing the decoder network
X = tf.keras.layers.UpSampling2D(name="upsample_1")(X)
X = tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='valid',activation='relu', name = 'conv_layer5')(X)
X = tf.keras.layers.UpSampling2D(name="upsample_2")(X)
X = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu', name = 'conv_layer6')(X)
X = tf.keras.layers.UpSampling2D(size=(3,3), name="upsample_3")(X)
X = tf.keras.layers.Conv2D(3, kernel_size=2, strides=1, padding='valid', activation='relu', name = 'conv_layer7')(X)
X = tf.keras.layers.UpSampling2D(name="upsample_4")(X)
decoder_output = tf.keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='valid', activation='relu', name = 'conv_layer8')(X)

# importing VGG-16 model as we use perceptual loss to train the autoencoder network
vgg16_model = VGG16(weights = 'imagenet')

for layer in vgg16_model.layers:
  layer.trainable = False

i = vgg16_model.get_layer('input_1').input
o1 = vgg16_model.get_layer('block2_conv1').output
o2 = vgg16_model.get_layer('block3_conv3').output
o3 = vgg16_model.get_layer('block4_conv2').output
o4 = vgg16_model.get_layer('block5_conv3').output

mod1 = Model(i, o1)
mod2 = Model(i, o2)
mod3 = Model(i, o3)
mod4 = Model(i, o4)


for layer in mod1.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same
    layer.trainable = False

for layer in mod2.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same
    layer.trainable = False

for layer in mod3.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same
    layer.trainable = False

for layer in mod4.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same
    layer.trainable = False


def custom_perceptual_loss(y_true, y_pred):
  m1 = mod1(y_true)
  m2 = mod2(y_true)
  m3 = mod1(y_true)
  m4 = mod2(y_true)

  n1 = mod1(y_pred)
  n2 = mod2(y_pred)
  n3 = mod1(y_pred)
  n4 = mod2(y_pred)

  l1 = tf.square(tf.subtract(m1, n1))
  l2 = tf.square(tf.subtract(m2, n2))
  l3 = tf.square(tf.subtract(m3, n3))
  l4 = tf.square(tf.subtract(m4, n4))

  _, a1, b1, c1 = m1.shape
  _, a2, b2, c2 = m2.shape
  _, a3, b3, c3 = m3.shape
  _, a4, b4, c4 = m4.shape
  s1 = a1*b1*c1
  s2 = a2*b2*c2
  s3 = a3*b3*c3
  s4 = a4*b4*c4
  s1 = tf.cast(s1, tf.float32)
  s2 = tf.cast(s2, tf.float32)
  s3 = tf.cast(s3, tf.float32)
  s4 = tf.cast(s4, tf.float32)

  loss = tf.sqrt(tf.reduce_sum(l1/s1) + tf.reduce_sum(l2/s2) + tf.reduce_sum(l3/s3) + tf.reduce_sum(l4/s4))

  return loss

autoencoder_model = Model(inputs = encoder_input, outputs = decoder_output)
autoencoder_model.compile(optimizer='adam', loss=custom_perceptual_loss)
h = autoencoder_model.fit(x=normal_images_train, y=normal_images_train, batch_size=batch_size_model, epochs=epoch_model, verbose=1)
