import datetime

import os
import numpy as np
import pandas as pd
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import image
from sklearn.model_selection import train_test_split

import ResNet
import utils

path = os.getcwd()
batch_size = 128


def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))


if __name__ == '__main__':
    # Loss Function...
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # ########### Data Initialization ###########
    # Reading Data.....
    train = pd.read_json(path + "/data/train.json")
    test = pd.read_json(path + "/data/test.json")
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    print("Loading Data done!")

    # Train data
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]
                             , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)   # Create a third channel
    X_angle_train = np.array(train.inc_angle)
    y_train = np.array(train["is_iceberg"])

    # Test data
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]
                             , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_test = np.array(test.inc_angle)

    # ############# Augmentation #################
    mean = nd.array([-20.655821, -26.320704, -23.488279])
    std = nd.array([5.200841, 3.3955173, 3.8151529])
    normalizer = image.ColorNormalizeAug(mean, std)
    flip = image.HorizontalFlipAug(1)

    X_train_new = [normalizer(nd.array(X_train[i])) for i in range(1604)]    # normalize
    X_train_new.extend([flip(X_train_new[i]) for i in range(1604)])     # flip
    y_train_new = np.append(y_train, y_train)  # y_train
    X_test_new = [normalizer(nd.array(X_test[i])) for i in range(8424)]  # X_test

    # Resize:
    X_train_new = [nd.transpose(X_train_new[i], (2, 0, 1)) for i in range(len(X_train_new))]
    X_test_new = [nd.transpose(X_test_new[i], (2, 0, 1)) for i in range(len(X_test_new))]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_train_new, y_train_new, test_size=0.2, random_state=66)

    # Aggregated Data_Set vs label, to pass to gluon data loader
    train_ds = [(X_train[i],y_train[i]) for i in range(len(y_train))]
    valid_ds = [(X_test[i],y_test[i]) for i in range(len(y_test))]
    train_valid_ds = [(X_train_new[i], y_train_new[i]) for i in range(len(y_train_new))]
    test_ds = [(X_test_new[i], 0) for i in range(len(X_test_new))]

    # Pass to gluon data loader
    loader = gluon.data.DataLoader
    train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
    valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
    train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
    test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

    # Get GPU information
    ctx = utils.try_gpu()

    # Train ResNet 18
    # CV based model validation pretty hard because of long training time.
    # Here we only choose one set to try.
    num_epochs = 1
    learning_rate = 0.05
    weight_decay = 1e-4
    lr_period = 25
    lr_decay = 0.1

    net = ResNet.get_net(ctx)
    net.hybridize()
    train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)

    # After get the parameters, we used all train_set to get the model, and to predict.
    net = ResNet.get_net(ctx)
    net.hybridize()
    train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)

    # prediction
    preds = []
    for data, label in test_data:
        output = net(data.as_in_context(ctx))
        preds.extend(output[:, 1].asnumpy())

    df = pd.DataFrame({'id': test['id'], 'is_iceberg': preds})
    df['id'] = df['id'].astype(str)
    df.to_csv('../submit/submission_test.csv', index=False)
