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
from matplotlib import pyplot
import matplotlib.pyplot as plt
import os

path_anomalous_read = r"/home/mazda/Documents/Natto Image Data/Anomalous Images/"
path_normal_read = r"/home/mazda/Documents/Natto Image Data/Normal Images/"

path_normal_save_train = r"/home/mazda/Documents/Natto Image Data/cnn/Normal Images-Resize Train/"
path_anomalous_save_test = r"/home/mazda/Documents/Natto Image Data/cnn/Anomalous Images-Resize Test/"
path_normal_save_test = r"/home/mazda/Documents/Natto Image Data/cnn/Normal Images-Resize Test/"

print("File Reading Started")
file_name_anomaly = np.array(os.listdir(path_anomalous_read), dtype=object)
file_name_normal = np.array(os.listdir(path_normal_read), dtype=object)

# shuffling them
np.random.shuffle(file_name_anomaly)
np.random.shuffle(file_name_normal)

# splitting the input data into train and test sets
size_images_train = 640
size_images_test = 127

# file_name_anomaloy_train = file_name_anomaly[:size_images_train]
file_name_normal_train = file_name_normal[:size_images_train]
file_name_anomaly_test = file_name_anomaly[size_images_train : size_images_train + 1 + size_images_test]
file_name_normal_test = file_name_normal[size_images_train : size_images_train + 1 + size_images_test]

# saving image files name that will be later used at loadmodel.py
print("saving image files name")
np.save("/home/mazda/Documents/Natto Image Data/cnn/ImageFilesNumpy/normalimagestrain.npy", file_name_normal_train)
np.save("/home/mazda/Documents/Natto Image Data/cnn/ImageFilesNumpy/anomalousimagestest.npy", file_name_anomaly_test)
np.save("/home/mazda/Documents/Natto Image Data/cnn/ImageFilesNumpy/normalimagestest.npy", file_name_normal_test)

# print(file_name_normal_test[:5])
# print(file_name_normal_test.shape)

resize_w_a = 224 # 480
resize_h_a = 224 # 640

resize_w_n = 224 # 480
resize_h_n = 224 # 640

no_of_channels = 3
print("reading image files ")
normal_train = []
for file in file_name_normal_train:
    f_img = path_normal_read + "/" + file
    img = Image.open(f_img)
    a, b = img.size
    img = img.resize((resize_w_n, resize_h_n))
    f_save = path_normal_save_train + "/" + file
    img.save(f_save)
    pix = np.array(img)
    normal_train.append(pix)

anomaly_test = []
for file in file_name_anomaly_test:
    f_img = path_anomalous_read + "/" + file
    img = Image.open(f_img)
    a, b = img.size
    img = img.resize((resize_w_a, resize_h_a))
    f_save = path_anomalous_save_test + "/" + file
    img.save(f_save)
    pix = np.array(img)	
    anomaly_test.append(pix)

normal_test = []
for file in file_name_normal_test:
    f_img = path_normal_read + "/" + file
    img = Image.open(f_img)
    a, b = img.size
    img = img.resize((resize_w_n, resize_h_n))
    f_save = path_normal_save_test + "/" + file
    img.save(f_save)
    pix = np.array(img)
    normal_test.append(pix)

normal_train = np.array(normal_train, dtype=object)
anomaly_test = np.array(anomaly_test, dtype=object)
normal_test = np.array(normal_test, dtype=object)

print(normal_train.shape)
print(anomaly_test.shape)
print(normal_test.shape)

print("saving as numpy arrays")
np.save("/home/mazda/Documents/Natto Image Data/cnn/ImagesNumpy/normalimagestrain.npy", normal_train)
np.save("/home/mazda/Documents/Natto Image Data/cnn/ImagesNumpy/anomalousimagestest.npy", anomaly_test)
np.save("/home/mazda/Documents/Natto Image Data/cnn/ImagesNumpy/normalimagestest.npy", normal_test)
