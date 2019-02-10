# -*- coding: utf-8 -*-
"""
I used some elements from:
 https://hackernoon.com/dl02-writing-a-neural-network-from-scratch-code-b32f4
 877c257

@author:
"""
import numpy as np


def onehot(target):
    n_classes = len(set(target))
    o = np.zeros(shape=(target.shape[0], n_classes))
    for i in range(target.shape[0]):
        o[i, int(target[i])] = 1
    return o


class Layer:
    """
    Layer object.
    """

    def __init__(self, n_hidden_minus, n_hidden, activation_function=None):

        # Value of the layer
        if n_hidden_minus == 0:
            self.feature = np.zeros((n_hidden, 1))
        else:
            self.feature = np.zeros((n_hidden, 1))
        # Weights for the layer
        # If zero then no act and weights.
        if n_hidden_minus == 0:
            self.layer_weights = None
            self.activation = None
            self.activation_function = None

        else:
            self.layer_weights = np.zeros((n_hidden_minus, n_hidden))
            self.activation = np.zeros([n_hidden, 1])
            self.activation_function = activation_function


class NN(object):

    def __init__(self, data, hidden_dims=(1024, 2048),
                 mode='train',
                 init_type='normal'):

        self.mode = mode
        self.init_type = init_type

        # Load data and normalize
        self.x_train = data['x_train'] / 255
        self.x_test = data['x_test'] / 255

        # For the target we use one hot encoding.
        self.target_train = onehot(data['t_train'])
        self.target_test = onehot(data['t_test'])
        #
        self.n_layers = len(hidden_dims) + 2
        self.init_type = init_type

        # Load data
        self.input_size = 784
        self.output_size = 10
        self.dims = (784,) + hidden_dims + (10,)

        # Layers initialization
        self.layers = []

        # Input layer
        layer_i = Layer(0, self.dims[0])
        self.layers.append(layer_i)

        for i in range(1, len(self.dims)):

            if i == len(self.dims)-1:
                act_fct = 'softmax'
            else:
                act_fct = 'relu'

            layer_i = Layer(self.dims[i-1], self.dims[i], act_fct)
            layer_i.layer_weights = self.initialize_weights(
                (self.dims[i-1],
                 self.dims[i]),
                init_type)

            # Add layer
            self.layers.append(layer_i)

    def initialize_weights(self, dims, init_type=None):
        """
        Weight initialization:
            -fully connected (with bias = 0) => (n_input) x (n_output)
             parameters.
        """
        if init_type == "normal":
            weights = np.random.normal(0, 1, size=dims)

        elif init_type == "zero":
            weights = np.zeros(dims)

        elif init_type == "glorot":
            n_in = dims[0]
            n_out = dims[1]
            d = np.sqrt(6 / (n_in + n_out))
            np.random.uniform(d, size=dims)
            weights = np.random.uniform(-d, d, size=dims)

        else:
            str_error = 'The type "{}" for weight initialization ' + \
                        'is not implemented'.format(init_type)
            raise NotImplementedError(str_error)

        return weights

    def forward(self, inputs, labels):
        """
        Forward pass.
        :param inputs: batch input
        :param labels: batch labels
        :return: average loss.
        """
        self.layers[0].feature = inputs

        for i in range(1, self.n_layers):

            temp = self.layers[i - 1].feature @ self.layers[i].layer_weights
            self.layers[i].activation = temp

            if self.layers[i].activation_function == 'relu':
                self.layers[i].feature = self.relu(temp)

            elif self.layers[i].activation_function == 'softmax':
                self.layers[i].feature = self.softmax(temp)

        # Compute loss and return it ?
        return self.loss(labels)

    def backward(self, labels):
        """
        Back Propagation: Follows algo 6.4 from DL book p.206
        """
        # Compute the gradient related to each sample

        # Output
        y = self.layers[-1].feature
        n_sample = y.shape[0]
        grad_dict = {layer: 0 for layer in range(self.n_layers)}

        for s in range(n_sample):

            # derivative of the loss with respect to softmax output
            # Will divide by zero ....
            # grad = - labels[s, :] / y[s, :]

            # Now with relu
            for i in range(self.n_layers - 1, 1, -1):

                feature = self.layers[i].feature[s, :]
                layer_activation = self.layers[i].activation_function

                # Small twist to avoid division by zero.
                if layer_activation == 'softmax':
                    grad = - labels[s, :] * (1 - feature)
                else:
                    grad_activation = self.grad_act_fct(feature,
                                                        layer_activation)
                    grad = grad * grad_activation

                # with respect to weights
                grad_w = grad * feature
                grad_dict[i] += grad_w / n_sample

                # Update gradient
                grad = self.layers[i].layer_weights @ grad

        return grad_dict

    def loss(self, labels):
        """
        Loss function: cross-entropy = -sum_c y * log(p)
        """
        return - np.sum(labels * np.log(self.layers[-1].feature))

    def update(self, grad, learning_rate=1e-1):
        """
        Parameters update
        """
        for i in range(self.n_layers - 1, 1, -1):
            self.layers[i].layer_weights -= learning_rate * grad[i]

    def train(self, batch_size=10000, learning_rate=0.1, n_epochs=10):
        """
        Train
        """
        loss_tracking = list()
        for epoch in range(n_epochs):

            epoch_loss = list()
            n_batch = self.x_train.shape[0] // batch_size
            #n_batch = 1

            for i in range(n_batch):

                sample_id = np.random.choice(range(self.x_train.shape[0]),
                                             size=batch_size, replace=False)

                # Batch data
                # xi = self.x_train[i * batch_size: (i + 1) * batch_size]
                # yi = self.target_train[i * batch_size: (i + 1) * batch_size]
                xi = self.x_train[sample_id, :]
                yi = self.target_train[sample_id, :]
                # Forward pass
                loss = self.forward(xi, yi)
                epoch_loss.append(loss)

                # Backward pass with weights update
                avg_grad = self.backward(yi)

                # print(avg_grad[3])
                # Update weights
                self.update(avg_grad, learning_rate)

                # Accuracy
                accuracy = self.check_accuracy(yi)

                print("Epoch: {}/{} - Minibatch {}/{}: "\
                      "/ Loss: {}"\
                      "/ Accuracy: {}".format(epoch+1, n_epochs, i, n_batch+1,
                                              loss, accuracy))

        # Average epoch loss
        loss_tracking.append(sum(epoch_loss) / len(epoch_loss))
        return loss_tracking, self.layers


    def check_accuracy(self, labels):
        """
        Look the the accuracy
        :param labels:
        :return:
        """
        a = self.layers[-1].feature
        counter = 0
        for i in range(a.shape[0]):
            pred = a[i, :]
            label_i = labels[i, :]
            if pred[np.where(label_i == 1)] == pred.max():
                counter += 1
        counter /= labels.shape[0]

        return counter

    def test(self):
        """
        Test algorithm
        """
        raise NotImplementedError('Method not used.')

    def grad_act_fct(self, feature, activation_function):
        """
        Compute the gradient of the activation function
        """
        if activation_function == 'relu':
            return self.grad_relu(feature)

        elif activation_function == 'softmax':
            return self.grad_softmax(feature)

    @staticmethod
    def grad_relu(x):
        """
        Relu derivative
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def grad_softmax(x):
        """
        Soft max jacobian
        :param x:
        :return: jacobian sum (vectro)
        """
        return x * (1 - x)

        # softmax_values = self.softmax(x)
        # jacobian_m = np.diag(softmax_values)
        # n, _ = jacobian_m.shape
        # for i in range(n):
        #     for j in range(n):
        #         if i == j:
        #             jacobian_m[i, j] = softmax_values[i] \
        #                                * (1 - softmax_values[i])
        #         else:
        #             jacobian_m[i, j] = -softmax_values[i] * softmax_values[j]
        # # return a vector that sums the jacobian
        # return jacobian_m.sum(axis=1)

    @staticmethod
    def relu(layer):
        """
        Activation function
        """
        # ReLU
        layer[layer < 0] = 0
        return layer

    @staticmethod
    def softmax(feature):
        """
        :param feature:
        :return: softmax output
        """
        # Max trick
        max_val = feature.max(axis=1)[:, None]

        feature_ = feature - max_val
        num = np.exp(feature_)
        denum = np.sum(num, axis=1)[:, None]

        return num / denum + 1e-64

