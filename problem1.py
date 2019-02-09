# -*- coding: utf-8 -*-
"""
I used some elements from:
 https://hackernoon.com/dl02-writing-a-neural-network-from-scratch-code-b32f4
 877c257

@author:
"""
import importlib
import mnist
import mlp
importlib.reload(mlp)

if __name__ == "__main__":

    mnist.init()

    x_train, t_train, x_test, t_test = mnist.load()

    mnist_data = {'x_train': x_train, 't_train': t_train,
                  'x_test': x_test, 't_test': t_test}

    mlp = mlp.NN(mnist_data, hidden_dims=(1024, 2048), mode='train',
                 init_type='normal')

    # for i in range(len(mlp.layers)):
    #     print("\n\nLayer: {}".format(i))
    #     print("\nFeature shape")
    #     print(mlp.layers[i].feature.shape)
    #     print("\nActivation function:")
    #     print(mlp.layers[i].activation_function)
    #     print("\nWeights shape:")
    #     if mlp.layers[i].layer_weights is not None:
    #         print(mlp.layers[i].layer_weights.shape)

    mlp.train(learning_rate=0.1)
    #
    # h0 = 784
    # h1 = 512
    # h2 = 256
    # h3 = 10
    # n_params = (h0 + 1) * h1 + (h1 + 1) * h2 + (h2 + 1) * h3
    # print("The number of parameters is: {}".format(n_params))
