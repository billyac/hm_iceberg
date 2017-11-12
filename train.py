import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from Models import Keras_CNN
import utils
import augmentors


if __name__ == '__main__':

    # Load Data
    train = pd.read_json("../data/train.json")
    test = pd.read_json("../data/test.json")
    print('Loading data......Done')

    # Pre-Process Data
    X_train, X_test, X_train_inc, X_test_inc, target_train, test_id = utils.preprocess_main(train, test)
    del train; del test; gc.collect();
    print('Pre-processing data........Done')

    # Train Test Split
    X_train_cv, X_valid, X_angle_train, X_angle_valid, y_train_cv, y_valid \
        = train_test_split(X_train, X_train_inc, target_train, random_state=6, train_size=0.75)

    # Pre-train preparation
    batch_size = 64
    file_path = "../weights_resnet1.hdf5"

    train_generator = augmentors.train_datagen_1.flow(X_train_cv, y_train_cv, batch_size=batch_size)
    validation_generator = augmentors.test_datagen_1.flow(X_valid, y_valid, batch_size=batch_size)

    # Load Model
    model = Keras_CNN.simple_resnet_v1()
    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # decay=0.0015) # Optimizer
    callbacks = utils.get_callbacks(filepath=file_path, patience=15, save_best=True)  # Callbacks
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # Compile
    model.fit_generator(train_generator,
                        steps_per_epoch=128,
                        epochs=200,
                        verbose=1,
                        validation_data=(X_valid, y_valid),
                        # validation_data = validation_generator,
                        # validation_steps = len(X_valid)/batch_size,
                        callbacks=callbacks)
    print('Fit Model.........Done')

    # Prior to submission
    model.load_weights(filepath=file_path)
    score = model.evaluate(X_valid, y_valid)
    print('Show score of model with valid set')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # if submit........
    print('Start prediction........')
    predicted_test = model.predict(X_test)
    submission = pd.DataFrame()
    submission['id'] = test_id
    submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))

    # Potential Leak angle:
    leaky_angle = [34.4721, 42.5591, 33.6352, 36.1061, 39.2340]
    mask = [test['inc_angle'][i] in leaky_angle for i in range(len(test))]
    column_name = 'is_iceberg'
    submission.loc[mask, column_name] = 1

    # submission to csv, need time string
    submission.to_csv('../submit/submission_time.csv', index=False)
