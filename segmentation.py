import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import cv2
import os
import pandas as pd
import sklearn.model_selection as skl
dir="Project Dataset"
classes=os.listdir(dir)
clr=[(0,0,0),(108,0,115),(145,1,122),(216,47,148),(254,246,242),(181,9,130),(236,85,157),(73,0,106),(127,255,255),(127,255,142),(255,127,127)]
def prep(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256,256])
    return image
def prep2(image,seg):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256,256])
    return image,seg
#To load the dataset,0=BCC,1=IEC,2=SCC
im=pd.read_excel('project.xlsx', usecols = [1,2,3],skiprows = 0).to_numpy()
#images=[]
mask=[]
train,test=skl.train_test_split(im,test_size=0.1,random_state=0)
for item in train:
    #images.append(cv2.imread(item[0])[:,:,::-1])
    mask.append(cv2.imread(item[1])[:,:,::-1])
#images=np.array(images,dtype=np.float32)/255
images=tf.constant(train[:,0])
mask=np.array(mask)
segmentation=np.zeros(shape=mask.shape[:-1],dtype=np.int32)
for i in range(len(clr)):
    segmentation[np.where(np.all(mask==np.array(clr[i]),axis=-1))]=i
#segmentation=tf.convert_to_tensor(segmentation)
#train_input=(images,segmentation)
ds=tf.data.Dataset.from_tensor_slices((images,segmentation))
ds=ds.map(prep2,num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)
train_labels,test_labels=train[:,-1],test[:,-1]
del images,mask,segmentation,i
#images=[]
mask=[]
for item in test:
    #images.append(cv2.imread(item[0])[:,:,::-1])
    mask.append(cv2.imread(item[1])[:,:,::-1])
#images=np.array(images,dtype=np.float32)/255
images=tf.constant(test[:,0])
mask=np.array(mask)
segmentation=np.zeros(shape=mask.shape[:-1],dtype=np.int32)
for i in range(len(clr)):
    segmentation[np.where(np.all(mask==np.array(clr[i]),axis=-1))]=i
#segmentation=tf.convert_to_tensor(segmentation)
#test_input=(images,segmentation)
test_ds=tf.data.Dataset.from_tensor_slices((images,segmentation))
test_ds=test_ds.map(prep2,num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)
del item,images,mask,segmentation,i,im
#Architecture will be U-net with convolution and convolution transpose along with skip connections, this is best for image segmentation
#This is functional API of tensorflow
img_inputs = keras.Input(shape=(256,256,3))
#Filters should be 3x3 with same colvolution,number of filters should be multiple of 32, there will be max pooling of 2x2 after evey conv.
#Batch normalization as well, Middle of model will be 16x16x256 after convolutions
x=layers.Conv2D(filters=16,kernel_size=3,padding="same")(img_inputs)
x=layers.BatchNormalization()(x)
x1=layers.LeakyReLU()(x)
x=layers.MaxPool2D(pool_size=2,strides=2)(x1)

x=layers.Conv2D(filters=32,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x2=layers.LeakyReLU()(x)
x=layers.MaxPool2D(pool_size=2,strides=2)(x2)

x=layers.Conv2D(filters=64,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x3=layers.LeakyReLU()(x)
x=layers.MaxPool2D(pool_size=2,strides=2)(x3)

x=layers.Conv2D(filters=128,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x4=layers.LeakyReLU()(x)
x=layers.MaxPool2D(pool_size=2,strides=2)(x4)

x=layers.Conv2D(filters=256,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(filters=256,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2DTranspose(filters=128,kernel_size=2,strides=2)(x)
x=layers.concatenate([x,x4],axis=-1)
x=layers.Conv2D(filters=128,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2DTranspose(filters=64,kernel_size=2,strides=2)(x)
x=layers.concatenate([x,x3],axis=-1)
x=layers.Conv2D(filters=64,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2DTranspose(filters=32,kernel_size=2,strides=2)(x)
x=layers.concatenate([x,x2],axis=-1)
x=layers.Conv2D(filters=32,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2DTranspose(filters=16,kernel_size=2,strides=2)(x)
x=layers.concatenate([x,x1],axis=-1)
x=layers.Conv2D(filters=16,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(filters=8,kernel_size=3,padding="same")(x)
x=layers.BatchNormalization()(x)
x=layers.LeakyReLU()(x)
out=layers.Conv2D(filters=len(clr),kernel_size=1,name="Segmented_Output")(x)
#x=layers.Conv2D(filters=8,kernel_size=3,strides=3)(out)
#x=layers.Flatten()(x)
#x=layers.Dense(8,activation='relu')(x)
#label=layers.Dense(3,activation='sigmoid',name="Label_Output")(x)
u_net=keras.Model(inputs=img_inputs, outputs=[out])#,label])
del img_inputs,out,x,x1,x2,x3,x4#,label
#u_net.summary()
loss=tf.keras.losses.SparseCategoricalCrossentropy()
optimizer=tf.keras.optimizers.Adam(learning_rate=0.003,
                                   beta_1=0.9,beta_2=0.98,epsilon=1e-9)
u_net.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])#, loss_weights=lossWeights)
del loss,optimizer

import math
def decaying_scheduler(epoch):
    """
        A step rate scheduler for setting the learning rate for each epoch.
        :param epoch: current epoch
        :return: learning rate for current epoch
        """
    learning_rate_min = 1e-6
    original_learning_rate_max = 1e-3
    original_cycle_length = 15
    epochs_per_cycle = original_cycle_length-(int(epoch/original_cycle_length))
    learning_rate_max = original_learning_rate_max/((int(epoch/original_cycle_length))+1)
    #print(learning_rate_max,learning_rate_min, epoch, epochs_per_cycle)

    return learning_rate_min + (learning_rate_max - learning_rate_min) * \
           (1 + math.cos(math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(decaying_scheduler, verbose=False)
model_save_path = 'models/'

file_path = os.path.join(model_save_path, 'm%i-%i.h5' % (2, 2))
checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(file_path, monitor= 'val_accuracy', verbose = 1, mode='max', save_best_only=True, save_freq='epoch')

my_callbacks = [checkpoints_callback,learning_rate_callback]
u_net.load_weights("Models/m2-2.h5")
ep=120
u_net.fit(ds,epochs=ep+10,validation_data=test_ds,initial_epoch=ep,callbacks=my_callbacks)
u_net.save_weights("Models/m2-2.h5")
rlc=np.array(clr)[:,::-1]
im=test
labels=im[:,2]
images=im[:,0]
ds=tf.data.Dataset.from_tensor_slices(im[:,0])
ds=ds.map(prep,num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
for i in range(len(labels)):
    st=images[i]
    images[i]=images[i][images[i].rfind("/")+1:]
l=0
for element in ds.as_numpy_iterator():
    output=u_net.predict(element)
    output=np.argmax(output,axis=-1)
    out=np.zeros(shape=(output.shape[0],256,256,3))
    for i in range(len(clr)):
        out[np.where(output==i)]=rlc[i]
    for i in range(output.shape[0]):
        j=(l*32)+i
        cv2.imwrite(f"Output/Test/{classes[labels[j]]}/{images[j]}",out[i,:,:,:])
    l=l+1
#To load the dataset,0=BCC,1=IEC,2=SCC
im=pd.read_excel('project.xlsx', usecols = [1,2,3],skiprows = 0).to_numpy()
_,test=skl.train_test_split(im,test_size=0.1,random_state=0)

mask=[]
for item in test:
    mask.append(cv2.imread(item[1])[:,:,::-1])
mask=np.array(mask)
y_true=np.zeros(shape=mask.shape[:-1],dtype=np.int32)
for i in range(len(clr)):
    y_true[np.where(np.all(mask==np.array(clr[i]),axis=-1))]=i
test_ds=tf.data.Dataset.from_tensor_slices(test[:,0])
test_ds=test_ds.map(prep,num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)
del item,mask,i,im,test
#Get confusion matrix of predicted output against correct output
y_pred=u_net.predict(test_ds)
y_pred=np.argmax(y_pred,axis=-1)
y_true=np.ravel(y_true)
y_pred=np.ravel(y_pred)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_true,y_pred,labels=list(range(len(clr))),normalize='all')
ConfusionMatrixDisplay.from_predictions(y_true,y_pred,labels=list(range(len(clr))))
