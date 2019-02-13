import numpy as np
import pickle


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        d = np.sqrt(6/(input_dim + output_dim))
        self.weights = np.random.uniform(-d, d, size=(input_dim, output_dim))# np.random.normal(0, 1, size=(input_dim, output_dim))  # np.zeros((input_dim, output_dim))
        self.biases = np.zeros(output_dim)  # np.random.normal(0, 1, size=output_dim)  # np.zeros(output_dim)
        self.latest_input = None
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, x):
        self.latest_input = x
        return np.matmul(x, self.weights) + self.biases

    def backward(self, grad):
        # store grad for weights and biases
        self.biases_grad = grad.sum(axis=0)
        self.weights_grad = np.matmul(self.latest_input.T, grad)
        return np.matmul(grad, self.weights.T)

    def step(self, lr):
        self.weights -= (self.weights_grad * lr / self.latest_input.shape[0])
        self.biases -= (self.biases_grad * lr / self.latest_input.shape[0])
        assert 1 == 1

    def get_params(self):
        return (self.weights, self.biases)

    def load_params(self, params):
        self.weights = params[0]
        self.biases = params[1]

    def __call__(self, x):
        return self.forward(x)


class Sigmoid:

    def __init__(self):
        self.latest_input = None
        self.sigmoid_op = np.vectorize(self._sigmoid_op)

    def forward(self, x):
        self.latest_input = x
        return self._sigmoid(x)

    def backward(self, grad):
        x = self._sigmoid(self.latest_input)
        return x * (1. - x) * grad  # element-wise multiplication

    def step(self, *args):
        pass

    def _sigmoid(self, x):
        return self.sigmoid_op(x)

    @staticmethod
    def _sigmoid_op(x):
        """ https://stackoverflow.com/questions/37074566/logistic-sigmoid-function-implementation-numerical-precision"""
        if x < 0:
            a = np.exp(x)
            return a / (1. + a)
        else:
            return 1. / (1. + np.exp(-x))

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x):
        return self.forward(x)


class ReLU:

    def __init__(self):
        self.latest_input = None

    def forward(self, x):
        if np.sum(x) == 0:
            pass
        self.latest_input = x
        x[x < 0] = 0.
        return x

    def backward(self, grad):
        # temp = np.clip(self.latest_input, 0., 1.)
        grad[self.latest_input <= 0] = 0.  # self.latest_input won't have negative values because x is being modified
        return grad

    def step(self, *args):
        pass

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x):
        return self.forward(x)


class CrossEntropyLoss:

    def __init__(self):
        self.latest_input = None
        self.labels = None

    def forward(self, x, labels):
        self.latest_input = x
        self.labels = labels
        # print('softmax: {}'.format(self._softmax(x)))
        return self._nll(self._softmax(x)[np.arange(len(x)), labels])

    def backward(self):
        x = self._softmax(self.latest_input)
        x[np.arange(len(x)), self.labels] -= 1
        return x

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1).reshape(-1, 1)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    @staticmethod
    def _nll(x):
        return -1. * np.log(x + np.finfo(float).eps)

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x, labels):
        return self.forward(x, labels)


class Softmax:

    def __init__(self):
        self.latest_input = None

    def forward(self, x):
        self.latest_input = x
        x = x - np.max(x, axis=1).reshape(-1, 1)
        return self._softmax(x)

    def backward(self, grad):
        x = self._softmax(self.latest_input)
        return x * grad - x * np.matmul(x, grad)

    def step(self, *args):
        pass

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1).reshape(-1, 1)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x):
        return self.forward(x)


class NLLLoss:

    def __init__(self):
        self.latest_input = None
        self.labels = None

    def forward(self, x, labels):
        self.latest_input, self.labels = x, labels
        return self._nll(x[labels])

    def backward(self):
        x = np.zeros(self.latest_input.shape)
        if self.latest_input[self.labels] == 0:
            self.latest_input[self.labels] = np.finfo(float).eps
        x[self.labels] = -1. / self.latest_input[self.labels]
        return x

    @staticmethod
    def _nll(x):
        return -1. * np.log(x + np.finfo(float).eps)

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x, labels):
        return self.forward(x, labels)


class NN:
    def __init__(self):
        # LinearLayer(100, 100),
        #                        Sigmoid(),
        self.layers = [LinearLayer(784, 1024), ReLU(), LinearLayer(1024, 256), ReLU(), LinearLayer(256, 10)]
        self.loss_fn = CrossEntropyLoss()
        self.latest_input = None
        self.latest_output = None
        self.lr = 0.25  # 0.0001 for ReLU; 0.001 for Sigmoid
        self.batch_size = 32

    def forward(self, x):
        self.latest_input = x
        for l in self.layers:
            x = l.forward(x)
        self.latest_output = x
        return x

    def backward(self, labels):
        loss = self.loss_fn(self.latest_output, labels)
        # print('latest out: {} labels: {}'.format(self.latest_output, labels))
        grad = self.loss_fn.backward()
        for l in reversed(self.layers):
            grad = l.backward(grad)
            # if grad.max() > 100:
                # print('large grad: {} max: {}'.format(grad, grad.max()))
        for l in self.layers:
            l.step(self.lr)
        return loss

    def train(self, n_epochs, train_data, train_labels, test_data, test_labels):
        train_inner_loops = train_data.shape[0] // self.batch_size
        for epoch in range(n_epochs):
            # permute data
            # permuted_indices = np.random.permutation(range(train_data.shape[0]))
            # train_data = train_data[permuted_indices]
            # train_labels = train_labels[permuted_indices]

            for iteration in range(train_inner_loops):
                train_inpt = train_data[iteration * self.batch_size: (iteration + 1) * self.batch_size] / 255.
                train_labl = train_labels[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                f = self.forward(train_inpt)
                b = self.backward(train_labl)
                if iteration % 200 == 0:
                    classification_accuracy = 1. - np.count_nonzero(f.argmax(axis=1) - train_labl) / float(self.batch_size)
                    print('epoch:{} iter: {} accuracy: {}'.format(epoch, iteration, classification_accuracy))
            self.test(test_data, test_labels, epoch)

    def test(self, test_data, test_labels, epoch=''):
        test_inner_loops = test_data.shape[0] // self.batch_size
        correct = 0.
        for iteration in range(test_inner_loops):
            test_inpt = test_data[iteration * self.batch_size: (iteration + 1) * self.batch_size] / 255.
            test_labl = test_labels[iteration * self.batch_size: (iteration + 1) * self.batch_size]
            f = self.forward(test_inpt)
            classification_accuracy = test_inpt.shape[0] - np.count_nonzero(f.argmax(axis=1) - test_labl)
            if np.sum(f.argmax(axis=1) - test_labl) != 0:
                temp = f.argmax(axis=1) - test_labl
                print('base: {} wrong indices: {}'.format(iteration * self.batch_size, temp.nonzero()))
            correct += classification_accuracy
        print('epoch: {} test accuracy is {}'.format(epoch, correct / test_data.shape[0]))

    def save(self, path):
        params = []
        for l in self.layers:
            params.append(l.get_params())
        with open(path, 'wb') as fp:
            pickle.dump(params, fp)

    def load(self, path):
        with open(path, 'rb') as fp:
            params = pickle.load(fp)
        assert len(params) == len(self.layers)
        for p, l in zip(params, self.layers):
            l.load_params(p)




nn = NN()
# Test XOR
# batch = [([0, 0], 0),
#          ([0, 1], 1),
#          ([1, 0], 1),
#          ([1, 1], 0)
#          ]
#
# for i in range(10000):
#     f = nn.forward(batch[i % 4][0])
#     b = nn.backward(batch[i % 4][1])
#     if i % 33 == 0:
#         print('input: {} output:{} loss:{}'.format(batch[i % 4], np.argmax(f), b))

from urllib import request
import gzip
import pickle

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


# init()
data = load()
print('training size: {} sample output:{}'.format(data[0].shape[0], data[1][0]))
# nn.train(11, *data)
# nn.save('/home/srini/PycharmProjects/IFT6135/params2.pkl')
nn.load('/home/srini/PycharmProjects/IFT6135/params.pkl')
# nn.test(data[2], data[3])
t_softmax = Softmax()
sample_idx = 430
weight_idx = 0
print('forward: {} label: {}'.format(t_softmax(nn.forward(np.array([data[2][sample_idx]]) / 255.)), data[3][sample_idx]))
nn.backward(np.array([data[3][sample_idx]]))
true_gradient = nn.layers[2].weights_grad[0][:10]
print('true gradient: {}'.format(true_gradient))
for base_eps in [1., 10., 100., 1000., 10000., 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]:
    for factor in range(1, 6):
        eps = 1./(factor * base_eps)
        diff_grads = []
        for weight_idx in range(10):
            # nn.test(data[2], data[3])
            nn.layers[2].weights[0][weight_idx] += eps
            nn.forward(np.array([data[2][sample_idx]]) / 255.)
            output_1 = nn.backward(np.array([data[3][sample_idx]]))
            nn.layers[2].weights[0][weight_idx] -= (2 * eps)
            nn.forward(np.array([data[2][sample_idx]]) / 255.)
            output_2 = nn.backward(np.array([data[3][sample_idx]]))
            diff_grads.append((output_1 - output_2)/(2 * eps))
            # print('true gradient: {} finite difference gradient: {}'.format(true_gradient, diff_grads[-1]))
            nn.layers[2].weights[0][weight_idx] += eps  # reset the weight to the old value
        print('eps:{} max diff: {}'.format(eps, np.abs(true_gradient - np.array(diff_grads)).max()))
