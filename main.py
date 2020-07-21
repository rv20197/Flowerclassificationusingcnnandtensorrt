#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:30:49 2020

@author: vatsal
"""


# =============================================================================
# %% Libraries
import os #To read the directory structure
import numpy as np #For some matrix math outside the tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.platform import gfile
from tensorflow.keras.preprocessing import image
import time
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Conv2D,Flatten,Dense,MaxPool2D
# =============================================================================

# =============================================================================
# %% Download Data
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
# =============================================================================

# =============================================================================
# %% Get train and validation data directories
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
# =============================================================================

# =============================================================================
# %% Data Augmentation

# config
img_width, img_height = 28,28 #width & height of input image
input_depth = 3 #1: gray image
train_data_dir = train_dir #data training path
testing_data_dir = val_dir #data testing path
epochs = 2 #number of training epoch
batch_size = 5 #training batch size

# define image generator for Keras,
# here, we map pixel intensity to 0-1
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# read image batch by batch
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    
    target_size=(img_width,img_height),#input image size
    batch_size=batch_size,#batch size
    class_mode='categorical')#categorical: one-hot encoding format class label

testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')
# =============================================================================

# =============================================================================
# %% Create a CNN Model
#define input image order shape
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

NUMB_FILTER_L1 = 20
NUMB_FILTER_L2 = 20
NUMB_FILTER_L3 = 20
NUMB_NODE_FC_LAYER = 10

#define the network
model = Sequential()

# Layer 1
model.add(Conv2D(NUMB_FILTER_L1, (5, 5), 
                 input_shape=input_shape_val, 
                 padding='same', name='input_tensor'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 2
model.add(Conv2D(NUMB_FILTER_L2, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 3
model.add(Conv2D(NUMB_FILTER_L3, (5, 5), padding='same'))
model.add(Activation('relu'))

#Layer 4
model.add(Conv2D(NUMB_FILTER_L3, (5, 5), padding='same'))
model.add(Activation('relu'))

# flattening the model for fully connected layer
model.add(Flatten())

# fully connected layer
model.add(Dense(NUMB_NODE_FC_LAYER, activation='relu'))
model.add(Dense(NUMB_NODE_FC_LAYER, activation='relu'))
model.add(Dense(NUMB_NODE_FC_LAYER, activation='relu'))

# output layer
model.add(Dense(train_generator.num_classes, 
                activation='softmax', name='output_tensor'))

# Compilile the network
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# Show the model summary
model.summary()
# =============================================================================

# =============================================================================
# %% fit model
# Fit the model
epochs = 30
model.fit_generator(
    train_generator,#our training generator
    #number of iteration per epoch = number of data / batch size
    steps_per_epoch=np.floor(train_generator.n/batch_size),
    epochs=epochs,#number of epoch
    validation_data=testing_generator,#our validation generator
    #number of iteration per epoch = number of data / batch size
    validation_steps=np.floor(testing_generator.n / batch_size))
model.save("/home/vatsal/Downloads/attachments/model.h5")
# =============================================================================

# =============================================================================
# %% Load Model
model = tf.keras.models.load_model('/home/vatsal/Downloads/attachments/model.h5')
model.summary()
# =============================================================================

# =============================================================================
# %% Convert keras model to tensorflow 
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, "/home/vatsal/Downloads/attachments/models/model")
# =============================================================================

# =============================================================================
# %% Read tensorflow model and convert to frozen model
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
    saver = tf.train.import_meta_graph("/home/vatsal/Downloads/attachments/models/model.meta")
    saver.restore(sess, "/home/vatsal/Downloads/attachments/models/model")
    your_outputs = ["output_tensor/Softmax"]
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),# graph+weight from the session
        output_node_names=your_outputs)
    with gfile.FastGFile("/home/vatsal/Downloads/attachments/models/frozen_model.pb", 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")
# =============================================================================

# =============================================================================
# %% Optimize the frozen model to TensorRT graph
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    max_batch_size=2,# specify your max batch size
    max_workspace_size_bytes=2*(10**9),# specify the max workspace
    precision_mode="FP16")

with gfile.FastGFile("/home/vatsal/Downloads/attachments/models/TensorRT_model.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")
# =============================================================================

# =============================================================================
# %% Count how many nodes/operations before and after optimization
# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)
# =============================================================================

# =============================================================================
# %% Function to read ".pb" model (TensorRT model is stored in ".pb")
def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def
# =============================================================================

# =============================================================================
# %%
input_img = image.load_img("/home/vatsal/.keras/datasets/flower_photos/tulips/11746548_26b3256922_n.jpg",
                           target_size=(28,28))
input_img = np.expand_dims( input_img, axis=0 )
input_img = input_img/255.0
# =============================================================================

# =============================================================================
# %% Perform inference using TensorRT model
TENSORRT_MODEL_PATH = '/home/vatsal/Downloads/attachments/models/TensorRT_model.pb'
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
        # read TensorRT model
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('input_tensor_input:0')
        output = sess.graph.get_tensor_by_name('output_tensor/Softmax:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 50
        out_pred = sess.run(output, feed_dict={input: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_tensorRT = total_time / n_time_inference
        print("average inference time: ", avg_time_tensorRT)
# =============================================================================

# =============================================================================
# %% Perform inference using the original tensorflow model
FROZEN_MODEL_PATH = '/home/vatsal/Downloads/attachments/models/frozen_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # read TensorRT model
        frozen_graph = read_pb_graph(FROZEN_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(frozen_graph, name='')
        input = sess.graph.get_tensor_by_name('input_tensor_input:0')
        output = sess.graph.get_tensor_by_name('output_tensor/Softmax:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 50
        out_pred = sess.run(output, feed_dict={input: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_original_model = total_time / n_time_inference
        print("average inference time: ", avg_time_original_model)
        print("TensorRT improvement compared to the original model:", avg_time_original_model/avg_time_tensorRT)
# =============================================================================
