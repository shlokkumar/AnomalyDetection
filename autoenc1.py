from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import os
import tensorflow.compat.v1 as tf
import math
from PIL import Image
from os import listdir
from os.path import isfile, join
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model
from keras.applications.vgg16 import VGG16
import time


print("reading training dataset from saved images arrays")
normal_images = np.load("/home/mazda/Documents/Natto Image Data/cnn/ImagesNumpy/normalimagestrain.npy", allow_pickle=True)
# normal_images = normal_images[:25]
print(normal_images.shape)
input_shape = normal_images.shape[1:]
print(input_shape)
def minmax(v):
  v_min = v.min(axis=(0, 1), keepdims=True)
  v_max = v.max(axis=(0, 1), keepdims=True)
  v = (v - v_min)/(v_max - v_min)
  return v

def mini_batches(X, mini_batch_size, shuffle = False):
    if shuffle:
        np.random.shuffle(X)
    m = X.shape[0]
    print(m)
    mini_batches = []
    num_complete_minibatches = math.floor(m/mini_batch_size)
    print(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batches.append(X[k * mini_batch_size:(k + 1) * mini_batch_size])
        
    return mini_batches

print("File Reading Ended")
	
display_step = 1000
lr=0.01
actf=tf.nn.relu
batchSize = 20
epochs = 2
normal_images = np.array(mini_batches(normal_images, batchSize, False))
print("Maximum: " + str(np.max(normal_images)) + ", Minimum: " + str( np.min(normal_images)))

print("Defining Network Started")
# designing the encoder network
input1 = Input(shape=(input_shape), name='Input_Layer')

conv_hid1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation='sigmoid', name='conv_layer1')(input1)
max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1")(conv_hid1)
bn1 = tf.keras.layers.BatchNormalization(name = "batch_norm1")(max_pool1)

conv_hid2 = tf.keras.layers.Conv2D(filters=28, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu", name="conv_layer2")(bn1)
max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2")(conv_hid2)
bn2 = tf.keras.layers.BatchNormalization(name = "batch_norm2")(max_pool2)

flatten = tf.keras.layers.Flatten(name="flatten_layer")(bn2)
bottom_layer = Dense(units=500, activation='relu', name="bottom_layer")(flatten)

# designing the decoder network
bottom_layer_reshape = tf.keras.layers.Reshape((25, 2, 10), input_shape=(500,), name="reshape_layer")(bottom_layer)

conv_hid3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(60, 60), strides=(1, 1), padding="valid", name="conv_layer3")(bottom_layer_reshape)
bn3 = tf.keras.layers.BatchNormalization(name = "batch_norm3")(conv_hid3)

conv_hid4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(66, 80), strides=(1, 1), padding="valid", name="conv_layer4")(bn3)
bn4 = tf.keras.layers.BatchNormalization(name = "batch_norm4")(conv_hid4)

output = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(76, 85), strides=(1, 1), padding="valid", name="conv_layer5")(bn4)
bn5 = tf.keras.layers.BatchNormalization(name = "batch_norm5")(output)

vgg16_model = VGG16(weights = 'imagenet')

for layer in vgg16_model.layers:
  layer.trainable = False

X_train = np.asarray(normal_images).astype('float32')
print(X_train.shape)

# this model maps an input to its reconstruction
ae_model = keras.Model(inputs = input1, outputs = bn5)
i = vgg16_model.get_layer('input_1').input
o1 = vgg16_model.get_layer('block1_conv2').output
o2 = vgg16_model.get_layer('block2_conv2').output
o3 = vgg16_model.get_layer('block3_conv3').output
o4 = vgg16_model.get_layer('block4_conv3').output

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

def custom_my_loss(y_true, y_pred):
  m1 = mod1(y_true)
  m2 = mod2(y_true)
  m3 = mod3(y_true)
  m4 = mod4(y_true)

  n1 = mod1(y_pred)
  n2 = mod2(y_pred)
  n3 = mod3(y_pred)
  n4 = mod4(y_pred)

  l1 = tf.square(tf.subtract(m1, n1))
  l2 = tf.square(tf.subtract(m2, n2))
  l3 = tf.square(tf.subtract(m3, n3))
  l4 = tf.square(tf.subtract(m4, n4))

  _, a1, b1, c1 = m1.shape
  _, a2, b2, c2 = m2.shape
  _, a3, b3, c3 = m3.shape
  _, a4, b4, c4 = m4.shape
  m1 = a1*b1*c1
  m2 = a2*b2*c2
  m3 = a3*b3*c3
  m4 = a4*b4*c4
  m1 = tf.cast(m1, tf.float32)
  m2 = tf.cast(m2, tf.float32)
  m3 = tf.cast(m3, tf.float32)
  m4 = tf.cast(m4, tf.float32)


  loss = tf.sqrt(tf.reduce_sum(l1/m1) + tf.reduce_sum(l2/m2) + tf.reduce_sum(l3/m3) + tf.reduce_sum(l4/m4))

  return loss

ae_model.compile(optimizer='sgd', loss=custom_my_loss)
print("Defining Network Ended")

print("Defining Network Training")
# h1 = autoencoder1.fit(new_normal_images_train, new_normal_images_train, epochs=epoch, batch_size=batchSize, shuffle=True)
counter = 0
store_all_losses = []
store_all_outputs = []
max_count = X_train.shape[0]
t1 = int(round(time.time() * 1000))
h1 = ae_model.fit(x=X_train[counter], y=X_train[counter], epochs=1, verbose=1, shuffle=True)
# h1 = h1.history['loss']

store_all_losses.append(h1.history['loss'][0])
store_all_outputs.append(ae_model.predict(X_train[counter]))
counter = counter + 1

for i in range(epochs):
  print("epoch: " + str(i+1) + "----------------------------------------->")
  if counter == max_count:
    counter = 0

  while counter < max_count:
    print("\tFor mini batch number: " + str(counter+1) + "-------------------->")
    h1 = ae_model.train_on_batch(x=X_train[counter], y=X_train[counter])
    output = ae_model.predict(X_train[counter])
    store_all_losses.append(h1)
    store_all_outputs.append(output)
    print(h1)
    counter = counter + 1

t2 = int(round(time.time() * 1000))
print(t1)
print(t2)
print(t2-t1)
store_all_losses = np.array(store_all_losses)
store_all_outputs = np.array(store_all_outputs)
print(store_all_losses.shape)
print(store_all_outputs.shape)

np.save("/home/mazda/Documents/Natto Image Data/cnn/Output/trainoutputimages.npy", store_all_outputs, allow_pickle=True)
np.save("/home/mazda/Documents/Natto Image Data/cnn/Output/trainoutputmetrics.npy", store_all_losses, allow_pickle=True)


# plot a loss curve w.r.t epochs

x = list(range(len(store_all_losses)))
plt.plot(x, store_all_losses)
plt.title('Model Loss')
plt.ylabel('Loss Values')
plt.xlabel('Steps')
plt.legend(['train'], loc='upper left')
plt.savefig('plot1.png', dpi=300, bbox_inches='tight')

