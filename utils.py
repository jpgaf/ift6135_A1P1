# -*- coding: utf-8 -*-
"""
MNIST download from https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
"""

import numpy as np
from urllib import request
import gzip
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    try:
        with open("mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    except FileNotFoundError:
        init()
        return load()


if __name__ == '__main__':
    init()


def plot(path):
    df = pd.read_csv(path, delimiter=' ', header=None, names=['variant', 'epoch', 'loss'])
    # df['value'] = 1. - df['value']
    filtered_df = df.groupby('variant').aggregate(np.mean)
    selected = filtered_df.sort_values('loss', ascending=False).head(10).index
    sns.set_palette(sns.color_palette("colorblind", 10))
    sns.relplot(x='epoch', y='loss', kind='line', hue='variant', data=df[df['variant'].isin(selected)])
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.show()


def gradient_checking(model, data):
    # t_softmax = Softmax()
    sample_idx = 430
    model.forward(np.array([data[2][sample_idx]]) / 255.)
    model.backward(np.array([data[3][sample_idx]]))
    true_gradient = model.layers[2].weights_grad[0][:10]
    print('true gradient: {}'.format(true_gradient))
    for base_eps in [1., 10, 100, 1e3, 1e4, 1e5]:
        for factor in range(1, 6):
            eps = 1. / (factor * base_eps)
            diff_grads = []
            for weight_idx in range(10):
                # nn.test(data[2], data[3])
                model.layers[2].weights[0][weight_idx] += eps
                model.forward(np.array([data[2][sample_idx]]) / 255.)
                output_1 = model.backward(np.array([data[3][sample_idx]]))
                model.layers[2].weights[0][weight_idx] -= (2 * eps)
                model.forward(np.array([data[2][sample_idx]]) / 255.)
                output_2 = model.backward(np.array([data[3][sample_idx]]))
                diff_grads.append((output_1 - output_2) / (2 * eps))
                # print('true gradient: {} finite difference gradient: {}'.format(true_gradient, diff_grads[-1]))
                model.layers[2].weights[0][weight_idx] += eps  # reset the weight to the old value
            print('eps:{} max diff: {}'.format(eps, np.abs(true_gradient - np.array(diff_grads)).max()))
