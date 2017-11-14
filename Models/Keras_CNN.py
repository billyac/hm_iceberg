from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model,load_model
from keras import initializers
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import layers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from keras import regularizers


def simple_resnet_v1(input_shape=(75, 75, 3), KernelSize = (5, 5), Momentum = 0.99):

    X_input = Input(input_shape)
    # input_CNN = ZeroPadding2D((0, 0))(X_input)
    input_CNN = BatchNormalization(momentum = Momentum)(X_input)

    # Input Layer
    input_CNN = Conv2D(32,kernel_size=KernelSize,padding='same', name='c11')(input_CNN)
    input_CNN = BatchNormalization(momentum = Momentum, name='b11')(input_CNN)
    input_CNN = Activation('elu')(input_CNN)
    input_CNN = MaxPooling2D((2,2),strides=(2, 2),name='m11')(input_CNN)
    # input_CNN = Dropout(0.25)(input_CNN)
    input_CNN = Conv2D(64,kernel_size=KernelSize,padding='same',name='c12')(input_CNN)
    input_CNN = BatchNormalization(momentum = Momentum,name='b12')(input_CNN)
    input_CNN = Activation('elu')(input_CNN)
    input_CNN = MaxPooling2D((2,2),strides=(2, 2),name='m12')(input_CNN)
    # input_CNN = Dropout(0.25)(input_CNN)

    # First Residual
    input_CNN_residual = BatchNormalization(momentum=Momentum)(input_CNN)
    input_CNN_residual = Conv2D(128,kernel_size=KernelSize,padding='same')(input_CNN_residual)
    input_CNN_residual = BatchNormalization(momentum = Momentum)(input_CNN_residual)
    input_CNN_residual = Activation('elu')(input_CNN_residual)
    # input_CNN_residual = Dropout(0.25)(input_CNN_residual)
    input_CNN_residual = Conv2D(64,kernel_size=KernelSize,padding='same')(input_CNN_residual)
    input_CNN_residual = BatchNormalization(momentum = Momentum)(input_CNN_residual)
    input_CNN_residual = Activation('elu')(input_CNN_residual)
    # input_CNN_residual = Dropout(0.25)(input_CNN_residual)

    input_CNN_residual = Add()([input_CNN_residual,input_CNN])

    ## Top CNN
    top_CNN = Conv2D(128, kernel_size = KernelSize, padding ='same')(input_CNN_residual)
    top_CNN = BatchNormalization(momentum=Momentum)(top_CNN)
    top_CNN = Activation('elu')(top_CNN)
    top_CNN = MaxPooling2D((2,2),strides=(2, 2))(top_CNN)
    top_CNN = Conv2D(256, kernel_size = KernelSize, padding ='same')(top_CNN)
    top_CNN = BatchNormalization(momentum=Momentum)(top_CNN)
    top_CNN = Activation('elu')(top_CNN)
    # top_CNN = Dropout(0.25)(top_CNN)
    top_CNN = MaxPooling2D((2,2),strides=(2, 2))(top_CNN)
    top_CNN = Conv2D(512, kernel_size = KernelSize, padding ='same')(top_CNN)
    top_CNN = BatchNormalization(momentum=Momentum)(top_CNN)
    top_CNN = Activation('elu')(top_CNN)
    # top_CNN = Dropout(0.25)(top_CNN)
    top_CNN = MaxPooling2D((2,2),strides=(2, 2))(top_CNN)
    top_CNN = GlobalMaxPooling2D()(top_CNN)

    # Dense Layers
    # X = Flatten()(top_CNN)
    X = Dense(512)(top_CNN)
    X = BatchNormalization(momentum=Momentum)(X)
    X = Activation('elu')(X)
    # X = Dropout(0.5)(X)
    X = Dense(256)(X)
    X = BatchNormalization(momentum=Momentum)(X)
    X = Activation('elu')(X)
    # X = Dropout(0.5)(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = X_input, outputs = X, name='simple_resnet')
    return model


def simple_resnet_1113(input_shape = (75,75,3), KernelSize = (5,5), Momentum = 0.99):

    X_input = Input(input_shape)
    #input_CNN = ZeroPadding2D((0, 0))(X_input)
    input_CNN = BatchNormalization(momentum = Momentum)(X_input)

    ## Input Layer
    input_CNN = Conv2D(32,kernel_size=KernelSize,padding='same', name='c11')(input_CNN)
    input_CNN = BatchNormalization(momentum = Momentum, name='b11')(input_CNN)
    input_CNN = Activation('elu')(input_CNN)
    input_CNN = MaxPooling2D((2,2),strides=(2, 2),name='m11')(input_CNN)
    #input_CNN = Dropout(0.25)(input_CNN)
    input_CNN = Conv2D(64,kernel_size=KernelSize,padding='same',name='c12')(input_CNN)
    input_CNN = BatchNormalization(momentum = Momentum,name='b12')(input_CNN)
    input_CNN = Activation('elu')(input_CNN)
    input_CNN = MaxPooling2D((2,2),strides=(2, 2),name='m12')(input_CNN)
    #input_CNN = Dropout(0.25)(input_CNN)

    ## First Residual
    input_CNN_residual = BatchNormalization(momentum=Momentum)(input_CNN)
    input_CNN_residual = Conv2D(128,kernel_size=KernelSize,padding='same')(input_CNN_residual)
    input_CNN_residual = BatchNormalization(momentum = Momentum)(input_CNN_residual)
    input_CNN_residual = Activation('elu')(input_CNN_residual)
    input_CNN_residual = Dropout(0.25)(input_CNN_residual)
    input_CNN_residual = Conv2D(64,kernel_size=KernelSize,padding='same')(input_CNN_residual)
    input_CNN_residual = BatchNormalization(momentum = Momentum)(input_CNN_residual)
    input_CNN_residual = Activation('elu')(input_CNN_residual)
    input_CNN_residual = Dropout(0.25)(input_CNN_residual)

    input_CNN_residual = Add()([input_CNN_residual,input_CNN])

    ## Top CNN
    top_CNN = Conv2D(128, kernel_size = KernelSize, padding ='same')(input_CNN_residual)
    top_CNN = BatchNormalization(momentum=Momentum)(top_CNN)
    top_CNN = Activation('elu')(top_CNN)
    top_CNN = MaxPooling2D((2,2),strides=(2, 2))(top_CNN)
    top_CNN = Conv2D(256, kernel_size = KernelSize, padding ='same')(top_CNN)
    top_CNN = BatchNormalization(momentum=Momentum)(top_CNN)
    top_CNN = Activation('elu')(top_CNN)
    top_CNN = Dropout(0.25)(top_CNN)
    top_CNN = MaxPooling2D((2,2),strides=(2, 2))(top_CNN)
    top_CNN = Conv2D(512, kernel_size = KernelSize, padding ='same')(top_CNN)
    top_CNN = BatchNormalization(momentum=Momentum)(top_CNN)
    top_CNN = Activation('elu')(top_CNN)
    top_CNN = Dropout(0.25)(top_CNN)
    top_CNN = MaxPooling2D((2,2),strides=(2, 2))(top_CNN)
    top_CNN = GlobalMaxPooling2D()(top_CNN)

    #Dense Layers
    #X = Flatten()(top_CNN)
    X = Dense(512)(top_CNN)
    X = BatchNormalization(momentum=Momentum)(X)
    X = Activation('elu')(X)
    X = Dropout(0.25)(X)
    X = Dense(256)(X)
    X = BatchNormalization(momentum=Momentum)(X)
    X = Activation('elu')(X)
    X = Dropout(0.25)(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = X_input, outputs = X, name='simple_resnet')
    return model
