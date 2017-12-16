import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, concatenate, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model,load_model
from keras import initializers
from keras.initializers import glorot_uniform
from keras.optimizers import Adam, RMSprop, rmsprop, SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import layers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
from time import localtime, strftime
import datetime
import gc

from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import load_model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def despatch(train_files, train_steps, train_batch_size, learning_rate):
    # Original Data
    train = pd.read_json(train-files)
    target_train=train['is_iceberg']
    # test = pd.read_json(eval_files)
    # test_id = test['id']

    # Train Set
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], 
                              X_band_2[:, :, :, np.newaxis],
                              ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
    X_train = X_train/100+0.5

    # incident angle:
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    X_train_inc = np.array(train.inc_angle)
    # X_test_inc = np.array(test.inc_angle)
    X_train_inc = X_train_inc / 60
    # X_test_inc_new = X_test_inc/60

    # Test Set
    # X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    # X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    # X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
    #                           , X_band_test_2[:, :, :, np.newaxis]
    #                          , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
    # X_test_new = X_test/100+0.5

    # del train, X_band_1, X_band_2, X_band_test_1, X_band_test_2, X_train, X_test, test
    # gc.collect()

    folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=16).split(X_train, target_train))

    prev_time = datetime.datetime.now()
    for j, (train_idx, valid_idx) in enumerate(folds):
        print('\n===================FOLD=',j+1)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_valid_cv = X_train[valid_idx]
        y_valid_cv = target_train[valid_idx]
        
        # Incidence Angle
        X_inc_cv = X_train_inc[train_idx]
        X_inc_valid = X_train_inc[valid_idx]

        #define file path and get callbacks
        # file_path = "../weights_1.hdf5"
        # callbacks = get_callbacks(filepath=file_path, patience=10)
        # Non-Trainable Layers
        model = getModel()
        #for layer in model.layers[:6]:
        #    layer.trainable = False
        # optimizer
        myoptim=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # compile
        model.compile(optimizer=myoptim, loss='binary_crossentropy', metrics=['accuracy'])
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_inc_cv, y_train_cv)
        
        model.fit_generator(gen_flow,
                            steps_per_epoch = 50,
                            epochs = 3,
                            shuffle = True,
                            verbose = 0,
                            validation_data = ([X_valid_cv,X_inc_valid], y_valid_cv))

        # #Getting the Best Model
        # model.load_weights(filepath=file_path)
        #Getting Training Score
        score = model.evaluate([X_train_cv,X_inc_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        #Getting Test Score
        score = model.evaluate([X_valid_cv,X_inc_valid], y_valid_cv, verbose=0)
        print('Validate loss:', score[0])
        print('Validate accuracy:', score[1])

        # #Getting validation Score.
        # pred_valid=model.predict([X_valid_cv,X_inc_valid])
        # y_valid_pred_log[valid_idx] = pred_valid.reshape(pred_valid.shape[0])

        # #Getting prediction
        # temp_test=model.predict([X_test_new, X_test_inc_new])
        # y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

        # #Getting Train Scores
        # temp_train=model.predict([X_train, X_train_inc])
        # y_train_pred_log+=temp_train.reshape(temp_train.shape[0])
        
        # del model
        # gc.collect()



def getModel():
    input_1 = Input(shape=(75,75,3), name = "image")
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    model_layer = ZeroPadding2D(padding=(32,32))(input_1)
    x = InceptionResNetV2(weights='imagenet', include_top=False, 
                 input_shape=(139,139,3), classes=1)(model_layer)
    #x = base_model.get_layer('block5_pool').output
    x = GlobalMaxPooling2D()(x)
    
    merge_one = concatenate([x, angle_layer])
    merge_one = Dense(512, name='fc2')(merge_one)
    merge_one = Activation('relu')(merge_one)
    merge_one = Dropout(0.20)(merge_one)
    merge_one = Dense(512, name='fc3')(merge_one)
    merge_one = Activation('relu')(merge_one)
    merge_one = Dropout(0.20)(merge_one)
    
    predictions = Dense(1, activation='sigmoid')(merge_one)
    
    model = Model(inputs=[input_1, input_2], outputs=predictions)

    return model



# Call back function
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


#Data Augmentation
batch_size = 32

# this is the augmentation configuration we will use for training
gen = ImageDataGenerator(
            rotation_range=20,  
            horizontal_flip=True,  
            vertical_flip=True,
            width_shift_range = 0.1,  
            height_shift_range = 0.1,  
            zoom_range = 0.1)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

def CrossValidation(X_train, X_train_inc, steps, learning_rate, decay, K=4):
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*target_train
    
    prev_time = datetime.datetime.now()
    for j, (train_idx, valid_idx) in enumerate(folds):
        print('\n===================FOLD=',j+1)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_valid_cv = X_train[valid_idx]
        y_valid_cv = target_train[valid_idx]
        
        # Incidence Angle
        X_inc_cv = X_train_inc[train_idx]
        X_inc_valid = X_train_inc[valid_idx]

        #define file path and get callbacks
        file_path = "../weights_1.hdf5"
        callbacks = get_callbacks(filepath=file_path, patience=10)
        # Non-Trainable Layers
        model = getModel()
        #for layer in model.layers[:6]:
        #    layer.trainable = False
        # optimizer
        myoptim=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        # compile
        model.compile(optimizer=myoptim, loss='binary_crossentropy', metrics=['accuracy'])
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_inc_cv, y_train_cv)
        
        model.fit_generator(
                            gen_flow,
                            steps_per_epoch = steps,
                            epochs = 100,
                            shuffle = True,
                            verbose = 0,
                            validation_data = ([X_valid_cv,X_inc_valid], y_valid_cv),
                            callbacks=callbacks)

        #Getting the Best Model
        model.load_weights(filepath=file_path)
        #Getting Training Score
        score = model.evaluate([X_train_cv,X_inc_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        #Getting Test Score
        score = model.evaluate([X_valid_cv,X_inc_valid], y_valid_cv, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Getting validation Score.
        pred_valid=model.predict([X_valid_cv,X_inc_valid])
        y_valid_pred_log[valid_idx] = pred_valid.reshape(pred_valid.shape[0])

        #Getting prediction
        temp_test=model.predict([X_test_new, X_test_inc_new])
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

        #Getting Train Scores
        temp_train=model.predict([X_train, X_train_inc])
        y_train_pred_log+=temp_train.reshape(temp_train.shape[0])
        
        del model
        gc.collect()

    y_test_pred_log=y_test_pred_log/K
    y_train_pred_log=y_train_pred_log/K

    #print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
    #print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    name_str = strftime("%Y%m%d%H%M", localtime())
    with open("../experiments/Output.txt", "a") as text_file:
        print("Submission: {}, Model Name: InceptionResNetV2 with Angle,4 fold, 0 level locked, no drop".format(name_str), file=text_file)
        print("Steps per epoch: {}, LR: {}, Decay: {}".format(steps,learning_rate,decay), file=text_file)
        print("Train Log Loss: {}".format(log_loss(target_train, y_train_pred_log)), file=text_file)
        print("Validation Log Loss: {}".format(log_loss(target_train, y_valid_pred_log)), file=text_file)
        print("Leader Board: _______________________________________", file=text_file)
        print("", file=text_file)
    submit(y_test_pred_log, name_str)
    
    cur_time = datetime.datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    
    print("Time used: "+ time_str)
    
    del y_test_pred_log, y_train_pred_log, temp_train, pred_valid, y_valid_pred_log
    gc.collect()


  
steps = [128]
lrs = [0.001]
decays = [0.01]
for step in steps:
    for lr in lrs:
        for decay in decays:
            CrossValidation(X_train_new, X_train_inc_new, step, lr, decay)