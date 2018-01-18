'''Trainning iceberg models on Google cloud'''
import argparse
import logging
import json
import os
from tensorflow.python.lib.io import file_io
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# from keras.applications.imagenet_utils import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
# from keras.initializers import glorot_uniform
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalMaxPooling2D, Input, MaxPooling2D,
                          ZeroPadding2D, concatenate)
# from keras.layers.advanced_activations import LeakyReLU, PReLU
# from keras.layers.merge import Concatenate
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam, RMSprop, rmsprop
# from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import layer_utils, plot_model
# from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

CV_RANDOM_SEED = 42
IMAGE_AUGMENT_SEED = 55


def dispatch(train_files, learning_rate, job_dir,
             train_batch_size=64, num_epochs=100, steps_per_epoch=15,
             cv=1, val_ratio=0.2, # cross validation
             decay=0.01, # learning rate decay
             fc_layers=[512], dropouts=[0.5], # fully connected layers
             trainable_layers=166, # trainable transfer learning model layers
            ):
    # log parameters.
    logging.info('start dispatch')
    logging.info('train_files: %s' %train_files)
    logging.info('learning_rate: %s' %learning_rate)
    logging.info('decay: %s' %decay)
    logging.info('job_dir: %s' %job_dir)
    logging.info('train_batch_size: %s' %train_batch_size)
    logging.info('num_epochs: %s' %num_epochs)
    logging.info('steps_per_epoch: %s' %steps_per_epoch)
    logging.info('cv: %s' %cv)
    logging.info('val_ratio: %s' %val_ratio)
    logging.info('fc_layers: %s' %fc_layers)
    logging.info('dropouts: %s' %dropouts)
    logging.info('trainable_layers: %s' %trainable_layers)

    # Original Data
    with file_io.FileIO(train_files[0], mode='r') as train_input:
        train_data = json.load(train_input)
    train_df = pd.DataFrame(train_data)
    train_target = train_df['is_iceberg']
    # TODO: add reading test data.

    # Preprocess
    # Images: resize images to 75*75, 2 channels, and scale each channel to
    # range 0 to 1.
    band_1 = np.array([
        np.array(band).astype(np.float64).reshape(75, 75)
        for band in train_df["band_1"]
    ])
    # Scale the input graph to -1 to 1
    band_1 = (band_1 - band_1.min()) / (band_1.max() - band_1.min()) * 2 - 1

    band_2 = np.array([
        np.array(band).astype(np.float64).reshape(75, 75)
        for band in train_df["band_2"]
    ])
    band_2 = (band_2 - band_2.min()) / (band_2.max() - band_2.min()) * 2 - 1

    X = np.concatenate(
        [
            band_1[:, :, :, np.newaxis],
            band_2[:, :, :, np.newaxis]
        ],
        axis=-1
    )

    # Incident angles: fill nan with 0, and scale to 0 - 1.
    train_df.inc_angle = train_df.inc_angle.replace('na', 0)
    X_inc = np.array(train_df.inc_angle)
    X_inc = X_inc / X_inc.max()


    # Set up cross validation: randomnly divide the data into several
    # training and validation splits, validation size can be set through
    # val_ratio, default to 20% of the total data.
    sample_size = len(train_target)
    validate_size = int(sample_size * val_ratio)
    np.random.seed(CV_RANDOM_SEED) # set random seed for reproducing results.
    folds = []
    for i in range(cv):
        # generate a shuffle.
        permutation = np.random.permutation(sample_size)
        # validation set.
        X_val = X[permutation[: validate_size]]
        X_inc_val = X_inc[permutation[: validate_size]]
        y_val = train_target[permutation[: validate_size]]
        # trainning set.
        X_train = X[permutation[validate_size :]]
        X_inc_train = X_inc[permutation[validate_size :]]
        y_train = train_target[permutation[validate_size :]]
        # add to folds.
        folds.append((X_train, X_inc_train, y_train, X_val, X_inc_val, y_val))

    # Training and cross validation.
    for i, (X_train, X_inc_train, y_train, X_val, X_inc_val, y_val) in enumerate(folds):
        logging.info('===================FOLD=%d' %i)
        # sanity check
        train_size = sample_size - validate_size
        assert len(X_train) == train_size
        assert len(X_inc_train) == train_size
        assert len(y_train) == train_size
        assert len(X_val) == validate_size
        assert len(X_inc_val) == validate_size
        assert len(y_val) == validate_size

        # TODO:
        # 1. save the best model
        # 2. predict on test set
        # 3. record prediction on trainning and validation for analysis

        model = get_model(fc_layers, dropouts, trainable_layers)

        # optimizer
        optimizer = Adam(
            lr=learning_rate, decay=decay,
            beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        )

        # compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # data flow generator, with image data augmented.
        # generator = ImageDataGenerator(
        #     horizontal_flip=True,
        #     vertical_flip=True
        # )
        generator = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            zoom_range = 0.1
        )

        gen_flow = gen_flow_for_two_inputs(
            X_train, X_inc_train, y_train, generator, train_batch_size
        )

        # TensorBoard callback, used to record training process for later
        # plotting using TensorBoard
        tensorboard = TensorBoard(
            log_dir=os.path.join(job_dir, 'logs'), write_graph=False
        )

        # Train model and validate along the way
        model.fit_generator(
            gen_flow,
            # TODO: investigate if the gen_flow shuffle before every epoch,
            # else, each epoch will be seeing the same samples
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            shuffle=True,
            verbose=1,
            validation_data=([X_val, X_inc_val], y_val),
            callbacks=[tensorboard])

def get_model(fc_layers, dropouts, trainable_layers=166):
    # dropouts either has the same size of fc_layers or has length 1 (dropout
    # ratio of all fc layers are the same in this case).
    assert len(dropouts) == 1 or len(fc_layers) == len(dropouts)
    # input layers
    input_image = Input(shape=(75, 75, 2), name="image")
    input_angle = Input(shape=[1], name="angle")

    # convert image layer to 3 channels and proper size so that it is compatible
    # the input of the transfer learning model.
    image_layer = Conv2D(filters=3, kernel_size=1, padding='same')(input_image)
    image_layer = ZeroPadding2D(padding=(32, 32))(image_layer)

    # transfer learning model
    transfer_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(139, 139, 3),
        pooling='avg')
    # freeze the transfer learning model layers except last layers, default to
    # last 166 layers for InceptionResNetV2.
    # the last 166 layers includes 10x block8 (Inception-ResNet-C block) of size
    # 8 x 8 x 2080 and one final convolution block of size 8 x 8 x 1536
    # When we make the transfer model a parameter, default trainable layers need
    # to be removed.
    if trainable_layers < 0:
        frozen_layers = []
    elif trainable_layers == 0:
        frozen_layers = transfer_model.layers
    else:
        frozen_layers = transfer_model.layers[:-trainable_layers]
    for layer in frozen_layers:
        layer.trainable = False
    # attach image layers to the input of the transfer model
    transfer_model = transfer_model(image_layer)

    # merge transfer learning model's bottleneck layer with angle input
    merged_model = concatenate([transfer_model, input_angle])

    # fully connected layers
    if len(dropouts) == 1:
        dropouts = dropouts * len(fc_layers)
    for i, (fc_layer, dropout) in enumerate(zip(fc_layers, dropouts)):
        merged_model = Dense(fc_layer, name='fc_%d' %i)(merged_model)
        merged_model = Activation('relu')(merged_model)
        merged_model = Dropout(dropout)(merged_model)

    # prediction layer
    predictions = Dense(1, activation='sigmoid')(merged_model)

    model = Model(inputs=[input_image, input_angle], outputs=predictions)

    return model

# Image Augmentation and generate training data flow

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and
# angle arrays
def gen_flow_for_two_inputs(X1, X2, y, generator, batch_size):
    gen_X1 = generator.flow(
        X1, y, batch_size=batch_size, seed=IMAGE_AUGMENT_SEED)
    gen_X2 = generator.flow(
        X1, X2, batch_size=batch_size, seed=IMAGE_AUGMENT_SEED)
    while True:
        X1i = gen_X1.next()
        X2i = gen_X2.next()
        # assert arrays are equal - this was for peace of mind, but slows down training
        np.testing.assert_array_equal(X1i[0],X2i[0])
        # yield batches of ([augmented images, angles], labels)
        yield [X1i[0], X2i[1]], X1i[1]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # set command line arguments to parse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-files',
        help='GCS or local paths to trainning data',
        nargs='+',
        required=True)

    parser.add_argument(
        '--num-epochs',
        help=""" Number of training data epochs on which to train. """,
        type=int,
        default=100
    )

    parser.add_argument(
        '--steps-per-epoch',
        help=""" Number of batches to go through per epoch. """,
        type=int,
        default=15
    )

    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=64
    )

    parser.add_argument(
        '--learning-rate',
        help='Learning rate',
        type=float,
        default=0.001
    )

    parser.add_argument(
        '--decay',
        help='Learning rate decay over each epoch',
        type=float,
        default=0.01
    )

    parser.add_argument(
        '--job-dir',
        help='Job dir',
        type=str,
        default=''
    )

    parser.add_argument(
        '--fc-layers',
        help='''
            Specify fully connected layers, provide a list of number, each
            represent the size of a layer. For example, "--fc-layers 512 256"
            means two fully connected layers of size 512 and 256.
        ''',
        type=int,
        nargs='*', # accept arbitrary numbers of input
        default=[512]
    )

    parser.add_argument(
        '--dropouts',
        help='''
            Specify fully connected layers' dropouts. When input multiple
            numnbers, need to be the same count as the fc-layers's input. Each
            number means the dropout ratio of the corresponding fully connected
            layer; when input one number, it will be the dropout ratio of all
            fc layers.
        ''',
        type=float,
        nargs='*', # accept arbitrary numbers of input
        default=[0.5] # default dropout for all fully connected layers
    )

    parser.add_argument(
        '--trainable-layers',
        help='''
            The number of last layers in the transfer learning model that is
            trainable. Default to 166 for InceptionResNetV2, need to change this
            when we make the transfer model a parameter. 0 means all layers are
            frozen; -1 (actually < 0) means all layers are trainable.
        ''',
        type=int,
        default=166 # default to all layers in transfer learning model are frozen
    )

    # parse command line arguments
    args = parser.parse_args()

    # start the training task
    dispatch(**args.__dict__)
