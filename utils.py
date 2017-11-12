import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def preprocess_average_third_band(df):

    # the third band is the average of 1st and 2nd band
    x_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    x_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    output = np.concatenate([x_band_1[:, :, :, np.newaxis], x_band_2[:, :, :, np.newaxis],
                             ((x_band_1+x_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

    return output


def preprocess_main(df_train, df_test):

    X_train = preprocess_average_third_band(df_train)
    X_test = preprocess_average_third_band(df_test)
    df_train.inc_angle = df_train.inc_angle.replace('na', 0)
    df_train.inc_angle = df_train.inc_angle.astype(float).fillna(0.0)
    X_train_inc = np.array(df_train.inc_angle)
    X_test_inc = np.array(df_test.inc_angle)
    target_train = df_train['is_iceberg']
    test_id = df_test['id']

    return X_train, X_test, X_train_inc, X_test_inc, target_train, test_id


def get_callbacks(file_path, patience=2, save_best=True):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(file_path, save_best_only=save_best)
    return [es, msave]
