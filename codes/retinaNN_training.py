###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import os
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD, Adam

from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

# ========= 从Config文件加载设置==================
config = configparser.RawConfigParser()
config.read('./configuration.txt')
# 修补数据集
path_data = config.get('data paths', 'path_local')

# 实验名称
name_experiment = config.get('experiment name', 'name')

# 训练设置
N_epochs = int(config.get('training settings', 'N_epochs'))

batch_size = int(config.get('training settings', 'batch_size'))

#============加载数据并分成补丁============================
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    # 仅在FOV内选择切片  (default == True)
    inside_FOV = config.getboolean('training settings', 'inside_FOV')
)

#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    # encoding path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)  # 32*48*48
    #
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)  # 64*24*24
    #
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)  # 128*12*12

    # decoding + concat path
    up1 = UpSampling2D(size=(2, 2))(conv3)  # 128*24*24
    up1 = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(up1)  # 64*24*24
    up1 = Dropout(0.2)(up1)
    up1 = Activation('relu')(up1)
    wg1 = Conv2D(32, (1, 1), padding='same', data_format='channels_first')(up1)  # 32*24*24
    wg1 = Dropout(0.2)(wg1)
    wx1 = Conv2D(32, (1, 1), padding='same', data_format='channels_first')(conv2)  # 32*24*24
    wx1 = Dropout(0.2)(wx1)
    psi1 = Activation('relu')(wg1 + wx1)  # 64*24*24
    psi1 = Conv2D(1, (1, 1), padding='same', data_format='channels_first')(psi1)  # 1*(64*24*24)
    psi1 = Dropout(0.2)(psi1)
    psi1 = Activation('sigmoid')(psi1)
    ag1 = psi1 * conv2
    up1 = Concatenate(axis=1)([ag1, up1])
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Conv2D(32, (3, 3), padding='same', data_format='channels_first')(up2)
    up2 = Dropout(0.2)(up2)
    up2 = Activation('relu')(up2)
    wg2 = Conv2D(16, (1, 1), padding='same', data_format='channels_first')(up2)
    wg2 = Dropout(0.2)(wg2)
    wx2 = Conv2D(16, (1, 1), padding='same', data_format='channels_first')(conv1)
    wx2 = Dropout(0.2)(wx2)
    psi2 = Activation('relu')(wg2 + wx2)
    psi2 = Conv2D(1, (1, 1), padding='same', data_format='channels_first')(psi2)
    psi2 = Dropout(0.2)(psi2)
    psi2 = Activation('sigmoid')(psi2)
    ag2 = psi2 * conv1
    up1 = Concatenate(axis=1)([ag2, up2])
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.3, nesterov=False)
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


"""
#====== 模型二 ============
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Dropout(0.2)(Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs))
    conv1 = Dropout(0.2)(Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(0.2)(Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1))
    conv2 = Dropout(0.2)(Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(0.2)(Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2))
    conv3 = Dropout(0.2)(Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(0.2)(Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool3))
    conv4 = Dropout(0.2)(Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(0.2)(Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool4))
    conv5 = Dropout(0.2)(Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5))

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Dropout(0.2)(Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(up6))
    conv6 = Dropout(0.2)(Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6))

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Dropout(0.2)(Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(up7))
    conv7 = Dropout(0.2)(Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7))

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Dropout(0.2)(Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up8))
    conv8 = Dropout(0.2)(Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8))

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Dropout(0.2)(Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up9))
    conv9 = Dropout(0.2)(Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9))
    outputs = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv9)
    print(outputs)
    outputs = core.Reshape((2, patch_height * patch_width))(outputs)
    outputs = core.Permute((2, 1))(outputs)
    outputs = core.Activation('sigmoid')(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    return model 
"""


"""
#====== 模型三 ============
def get_unet(n_ch,patch_height,patch_width):
    k = 3  # kernel size
    s = 2  # stride
    img_ch = n_ch  # image channels
    out_ch = 1  # output channel
    img_height = patch_height
    img_width = patch_width
    n_filters = 32
    padding = 'same'

    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(n_filters, (k, k), padding=padding, data_format='channels_first')(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding=padding, data_format='channels_first')(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding, data_format='channels_first')(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, data_format='channels_first')(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, data_format='channels_first')(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, data_format='channels_first')(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Concatenate(axis=1)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, data_format='channels_first')(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=1)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, data_format='channels_first')(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Concatenate(axis=1)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding, data_format='channels_first')(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding, data_format='channels_first')(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=1)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding, data_format='channels_first')(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding=padding, data_format='channels_first')(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(2, (1, 1), padding=padding, activation='relu', data_format='channels_first')(conv9)
    outputs = core.Reshape((2, patch_height * patch_width))(outputs)
    outputs = core.Permute((2, 1))(outputs)
    outputs = core.Activation('sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
"""


#========= 保存您正在为神经网络提供的样本 ==========
N_sample = min(patches_imgs_train.shape[0],40)

visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'../'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'../'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== 构建并保存模型体系结构 =====
n_ch = patches_imgs_train.shape[1]                  # 灰色图像，通道数为1
patch_height = patches_imgs_train.shape[2]          # 长为48
patch_width = patches_imgs_train.shape[3]           # 宽为48

model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
print(model.output_shape)


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('../'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='../'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)#save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # 减少内存消耗
model.fit(patches_imgs_train,
          patches_masks_train,
          epochs = N_epochs,
          batch_size = batch_size,
          verbose = 1,
          shuffle=True,
          validation_split = 0.1,
          callbacks = [checkpointer])

#========== Save and test the last model ===================
model.save_weights('../'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
