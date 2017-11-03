import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon


def read_jason(file='', loc='data'):
    df = pd.read_json('{}/{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2), axis=-1)
    del band1, band2
    return df, bands


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def accuracy(output, label):
    return np.mean(output.asnumpy().argmax(axis=1) == label.asnumpy())


def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.array([np.sum(net(X).asnumpy().argmax(axis=1) == y.asnumpy())]).copyto(mx.cpu())
        acc.wait_to_read()  # don't push too many operators into backend
        n += batch_size
    return acc.asscalar() / n


def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])
