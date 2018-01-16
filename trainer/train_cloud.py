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


def dispatch(train_files, learning_rate, job_dir, train_batch_size=32,
             cv=1, val_ratio=0.2, # cross validation
             decay=0.01, # learning rate decay
            ):
    # log some infomation.
    logging.info("start dispatch")
    logging.info(str(train_files))

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
    # Scale the input graph to 0-1
    band_1 = (band_1 - band_1.min()) / (band_1.max() - band_1.min())

    band_2 = np.array([
        np.array(band).astype(np.float64).reshape(75, 75)
        for band in train_df["band_2"]
    ])
    band_2 = (band_2 - band_2.min()) / (band_2.max() - band_2.min())

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

        # TODO: try freeze the transfer model layers.
        model = get_model()

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
        generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True
        )

        gen_flow = gen_flow_for_two_inputs(
            X_train, X_inc_train, y_train, generator, train_batch_size
        )

        # TensorBoard callback, used to record training process for later
        # plotting using TensorBoard
        tensorboard = TensorBoard(log_dir=os.path.join(job_dir, 'logs'), write_graph=False)

        # Train model and validate along the way
        model.fit_generator(
            gen_flow,
            # TODO: investigate if the gen_flow shuffle before every epoch,
            # else, each epoch will be seeing the same samples
            steps_per_epoch=15,
            epochs=50,
            shuffle=True,
            verbose=1,
            validation_data=([X_val, X_inc_val], y_val),
            callbacks=[tensorboard])

def get_model():
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
        pooling='max')
    # freeze the transfer learning model layers
    for layer in transfer_model.layers:
        layer.trainable = False
    # attach image layers to the input of the transfer model
    transfer_model = transfer_model(image_layer)

    # merge transfer learning model's bottleneck layer with angle input
    merged_model = concatenate([transfer_model, input_angle])
    # fully connected layers
    merged_model = Dense(512, name='fc1')(merged_model)
    merged_model = Activation('relu')(merged_model)
    merged_model = Dropout(0.5)(merged_model)

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

    # parser.add_argument(
    #     '--num-epochs',
    #     help="""\
    #         Maximum number of training data epochs on which to train.
    #         If both --max-steps and --num-epochs are specified,
    #         the training job will run for --max-steps or --num-epochs,
    #         whichever occurs first. If unspecified will run for --max-steps.\
    #     """,
    #     type=int,
    # )

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
        '--job-dir',
        help='Job dir',
        type=str,
        default=''
    )

    # parse command line arguments
    args = parser.parse_args()

    # start the training task
    dispatch(**args.__dict__)
