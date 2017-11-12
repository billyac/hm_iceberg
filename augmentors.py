from keras.preprocessing.image import ImageDataGenerator


train_datagen_1 = ImageDataGenerator(  # rescale=1./50,
                                     rotation_range=20,  horizontal_flip=True,  vertical_flip=True,
                                     width_shift_range=0.30,  height_shift_range=0.30,  zoom_range=0.1)


test_datagen_1 = ImageDataGenerator(  # rescale=1./50,
                                    horizontal_flip=True, vertical_flip=True)
