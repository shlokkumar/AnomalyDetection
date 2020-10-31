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
from sklearn.preprocessing import MinMaxScaler
import os

print("reading training dataset from saved images arrays")
new_normal_images_train = np.load("/home/mazda/OneDrive/Ono Project/Anomaly Detection/Code Base/Shlok /ImagesNumpy/normalimagestrain.npy")
print(new_normal_images_train.shape)

h = 216
w = 288
c = 3

scaler = MinMaxScaler()
new_normal_images_train = scaler.fit_transform(new_normal_images_train)

print("File Reading Ended")
# 480*640*3 = 921600
# 120*160*3 = 57600 h*w*c
# 216*288*3 = 186624 h*w*c
#     w1, b1    w2, b2   w3, b3  w4, b4  w5, b5  w6, b6
# i/p   ->  hid1 ->  hid2 -> hid3 ->  hid4 ->  hid5 ->  o/p
# 57600 -> 25000 -> 10000 -> 2000 -> 10000 -> 25000 -> 57600
# designing input, hidden and outpur layers and other hyper-paramters
input_nodes = h*w*c
num_hid1 = 10000 
num_hid2 = 5000
num_hid3 = 2000
num_hid4 = 30000
output_nodes =  input_nodes
display_step = 1000
lr=0.01
actf=tf.nn.relu
batchSize = 64
epochs = 200
print("Maximum: " + str(np.max(new_normal_images_train)) + ", Minimum: " + str( np.min(new_normal_images_train)))

print("Defining Network Started")
# defining encoder n/w
input_data = keras.Input(shape=(input_nodes,))
hid1 = layers.Dense(num_hid1, activation='sigmoid')(input_data)
hid2 = layers.Dense(num_hid2, activation='relu')(hid1)
hid3 = layers.Dense(num_hid3, activation='relu')(hid2)
#defining decoder n/w
hid4 = layers.Dense(num_hid4, activation='relu')(hid3)
output_data = layers.Dense(output_nodes, activation='relu')(hid4)

# this model maps an input to its reconstruction
autoencoder1 = keras.Model(input_data, output_data)

print("Defining Network Ended")

autoencoder1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=tf.keras.losses.MeanSquaredError())

print("Defining Network Training")
h1 = autoencoder1.fit(new_normal_images_train, new_normal_images_train, epochs=epochs, batch_size=batchSize, shuffle=True)

losses = h1.history['loss'][0:]
np.save("/home/mazda/Documents/Output/lossplot.npy", losses, allow_pickle=True)

"""x = list(range(len(losses)))
plt.plot(x, losses)
plt.grid(True,which="both", linestyle='--')
plt.title('Model Loss')
plt.ylabel('Loss Values')
plt.xlabel('Steps')
plt.yscale("log", base = 10)
plt.legend(['train'], loc='upper left')
plt.savefig('plot1_loss_fit.png', dpi=300, bbox_inches='tight')	"""
