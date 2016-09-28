#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com


from __future__ import print_function

#import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input,merge,Convolution2D,MaxPooling2D,UpSampling2D,Dropout,BatchNormalization,Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras import backend as K

from data import load_train_data,load_test_data

img_rows = 64
img_cols = 80
smooth = 1.

'''
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0],imgs.shape[1],img_rows,img_cols),dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i,0] = cv2.resize(imgs[i,0],(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    return imgs_p
'''

def dice_coef(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
#    intersection = float(len(np.where(y_true_f==y_pred_f)))
#    return (intersection+0.5)/(y_true_f.shape[0])
    return (2.*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)

def create_model():
    inputs = Input((1,img_rows,img_cols))
    conv1 = Convolution2D(32,3,3,border_mode='same')(inputs)
    conv1 = Convolution2D(32,3,3,border_mode='same')(conv1)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation(activation='relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2,2))(act1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Convolution2D(64,3,3,border_mode='same')(drop1)
    conv2 = Convolution2D(64,3,3,border_mode='same')(conv2)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation(activation='relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2,2))(act2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = Convolution2D(128,3,3,border_mode='same')(drop2)
    conv3 = Convolution2D(128,3,3,border_mode='same')(conv3)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation(activation='relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2,2))(act3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = Convolution2D(256,3,3,border_mode='same')(drop3)
    conv4 = Convolution2D(256,3,3,border_mode='same')(conv4)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation(activation='relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2,2))(act4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = Convolution2D(512,3,3,border_mode='same')(drop4)
    conv5 = Convolution2D(512,3,3,border_mode='same')(conv5)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation(activation='relu')(bn5)
    drop5 = Dropout(0.5)(act5)

    up6 = merge([UpSampling2D(size=(2,2))(drop5),act4],mode='concat',concat_axis=1)
    conv6 = Convolution2D(256,3,3,border_mode='same')(up6)
    conv6 = Convolution2D(256,3,3,border_mode='same')(conv6)
    bn6 = BatchNormalization()(conv6)
    act6 = Activation(activation='relu')(bn6)
    drop6 = Dropout(0.5)(act6)

    up7 = merge([UpSampling2D(size=(2,2))(drop6),act3],mode='concat',concat_axis=1)
    conv7 = Convolution2D(128,3,3,border_mode='same')(up7)
    conv7 = Convolution2D(128,3,3,border_mode='same')(conv7)
    bn7 = BatchNormalization()(conv7)
    act7 = Activation(activation='relu')(bn7)
    drop7 = Dropout(0.5)(act7)

    up8 = merge([UpSampling2D(size=(2,2))(drop7),act2],mode='concat',concat_axis=1)
    conv8 = Convolution2D(64,3,3,border_mode='same')(up8)
    conv8 = Convolution2D(64,3,3,border_mode='same')(conv8)
    bn8 = BatchNormalization()(conv8)
    act8 = Activation(activation='relu')(bn8)
    drop8 = Dropout(0.5)(act8)

    up9 = merge([UpSampling2D(size=(2,2))(drop8),act1],mode='concat',concat_axis=1)
    conv9 = Convolution2D(32,3,3,border_mode='same')(up9)
    conv9 = Convolution2D(32,3,3,border_mode='same')(conv9)
    bn9 = BatchNormalization()(conv9)
    act9 = Activation(activation='relu')(bn9)
    drop9 = Dropout(0.5)(act9)

    conv10 = Convolution2D(1,1,1,activation='sigmoid')(drop9)

    model = Model(input=inputs,output=conv10)

#    model.load_weights("./generated_data/model_full_training_2.hdf5")
    model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss,metrics=[dice_coef])
    return model

def train_and_predict():
    '''    
    print("-"*30)
    print("Loading train data...")
    print('-'*30)
    
#    train_images,train_masks = load_train_data()
    
#   train_images = preprocess(train_images)
#   train_masks = preprocess(train_masks)
    
#    np.save("./generated_data/train_images_64*80.npy",train_images)
#    np.save("./generated_data/train_masks_64*80.npy",train_masks)

    train_images = np.load("./generated_data/train_images_concate_4_64*80.npy")
    train_masks = np.load("./generated_data/train_masks_concate_4_64*80.npy")
    
    train_images = train_images.astype('float32')
    mean  = np.mean(train_images)
    std = np.std(train_images)

    train_images -= mean
    train_images /= std

    train_masks = train_masks.astype('float32')
    train_masks /= 255.

    print('-'*30)
    print('Creating and compile model...')
    print('-'*30)
    '''
    model = create_model()
    model_checkpoint = ModelCheckpoint('./generated_data/model_full_training_2.hdf5',monitor='loss',save_best_only=True)
    stop1 = EarlyStopping(monitor='loss',patience=3,mode='min')
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')
    '''
    print('-'*30)
    print('fitting mdoel...')
    print('-'*30)
    model.fit(train_images,train_masks,batch_size=240,nb_epoch=22,verbose=1,shuffle=True,callbacks=[model_checkpoint,stop1])
    
    print('-'*30)
    print('Loading test data...')
    print('-'*30)
    
    test_images,test_ids = load_test_data()
#    test_images = preprocess(test_images)
    
#    np.save("./generated_data/test_images_64*80.npy",test_images)
    
    test_images = test_images.astype('float32')
    test_images -= mean
    test_images /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    test_masks = model.predict(test_images,verbose=1)
    np.save('./generated_data/test_masks_full_training_2.npy',test_masks)
    '''

if __name__ == '__main__':
    train_and_predict()
