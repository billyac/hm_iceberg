import datetime
import os

import numpy as np
import pandas as pd
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from sklearn.model_selection import train_test_split

import ResNet
import utils

base_path = os.path.join('..', 'data')


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


def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_df['inc_angle'].replace('na', np.nan, inplace=True)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'], c_row['band_2']], -1).reshape((75, 75, 2))], 1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


if __name__ == '__main__':
    # Loss Function...
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # Reading Data.....
    train_df, train_images = load_and_format(os.path.join(base_path, 'train.json'))
    print('training', train_df.shape, 'loaded', train_images.shape)
    test_df, test_images = load_and_format(os.path.join(base_path, 'test.json'))
    print('testing', test_df.shape, 'loaded', test_images.shape)

    # Transpose from 75*75*2 -> 2*75*75
    train_images = [nd.transpose(nd.array(train_images[x]),(2,0,1)) for x in range(len(train_df['is_iceberg']))]
    test_images = [nd.transpose(nd.array(test_images[x]),(2,0,1)) for x in range(len(test_df['id']))]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(train_images, np.asarray(train_df['is_iceberg']), test_size=0.2, random_state=42)

    # Aggregated Data_Set vs label
    train_ds = [(X_train[i],y_train[i]) for i in range(len(y_train))]
    valid_ds = [(X_test[i],y_test[i]) for i in range(len(y_test))]
    train_valid_ds = [(train_images[i], np.asarray(train_df['is_iceberg'][i])) for i in range(len(train_df['is_iceberg']))]
    test_ds = [(test_images[i], 0) for i in range(len(test_df['id']))]

    # Pass to gluon data loader
    batch_size = 128
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
    num_epochs = 50
    learning_rate = 0.1
    weight_decay = 1e-4
    lr_period = 20
    lr_decay = 0.1

    net = ResNet.get_net(ctx)
    net.hybridize()
    train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)

    # After get the parameters, we used all train_set to get the model, and to predict.
    net = ResNet.get_net(ctx)
    net.hybridize()
    train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)

    # prediction
    preds = []
    for data, label in test_data:
        output = net(data.as_in_context(ctx))
        preds.extend(output.asnumpy().argmax(axis=1))

    df = pd.DataFrame({'id': test_df['id'], 'is_iceberg': preds})
    df['id'] = df['id'].astype(str)
    df.to_csv('../submit/submission2.csv', index=False)
