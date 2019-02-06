# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:20:43 2019

@author: JPGagnonFleury
"""

""" 
https://hackernoon.com/dl02-writing-a-neural-network-from-scratch-code-b32f4877c257
""" 

import numpy as np


class layer:
    def __init__(self, n_in, n_out, activation_function=None):
       
        # Value of the layer
        self.feature = np.zeros([n_in, 1])
        
        #  Weights for the layer
        if  n_in != 0:
            self.layer_weights = np.zeros((n_in, n_out))
            self.activation = np.zeros([n_out, 1]) 
            self.activation_function = activation_function

        else:
            self.layer_weights = None
            self.activation = None
            self.activation_function = None
         

class NN(object):
    
    def __init__(self, hidden_dims=(1024, 2048), mode='train',
                 init_type='zero', datapath=None, model_path=None):
        
        self.n_layers = len(hidden_dims) + 2
        self.init_type=init_type
        
        # Load data
        self.input_size = 784
        self.output_size = 10
        self.dims = (0, 784) + hidden_dims + (10,)
        
        # Layers initialization
        self.layers = []
        for i in range(1, len(self.dims)):
            
            if i == 0 :
                layer_i = layer(0, self.dims[i])
                
            else:
                
                if i == 4:
                    act_fct = 'softmax'
                else:
                    act_fct = 'relu'
                    
                layer_i = layer(self.dims[i-1], self.dims[i], act_fct)
                layer_i.layer_weights = self.initialize_weights((self.dims[i-1], 
                                                                 self.dims[i]), 
                                                                init_type)            
                
            # Add layer
            self.layers.append(layer_i)    

        
    def initialize_weights(self, dims, init_type=None):
        """
        Weight initializaiton: 
            -fully connected (with bias = 0) => (n_input) x (n_output) 
             parameters.
        """
        if init_type == 'normal':
            weights = np.random.normal(0, 1, size=dims)
            
        elif init_type == 'zero':
            weights = np.zeros(dims)  
            
        else:
            str_error = 'The type "{}" for weight initialization ' + \
                        'is not implemented'.format(init_type)
            raise NotImplementedError(str_error)
            
        return weights
        
    
    def forward(self, inputs, labels):
        """
        Forward pass
        """
        self.layers[0].feature = inputs
        
        for i in range(1, self.num_layers):
            
            temp = np.matmult(self.layers[i-1].feature, 
                              self.layers[i].layer_weights)
            self.layers[i].activation = temp
            
            if self.layers[i].activation_function == 'relu':
                self.layers[i].feature = self.relu(temp)
            
            elif self.layers[i].activation_function == 'softmax':
                self.layers[i].feature = self.softmax(temp)
        
        # Compute loss and return it ? 
        return self.loss(self.layers[-1].feature, labels)


    def relu(self, inputs):
        """
        Activation function
        """
        # ReLU
        layer[layer < 0] = 0
        return layer        
           
        
    def softmax(self, feature):
        """
        Softmax
        """
        # TO DO: ADD MAX TRIKC
        exp = np.exp(layer)
        
        if isinstance(layer[0], np.ndarray):
            return exp / np.sum(exp, axis=1, keepdims=True)
        
        else:
            return exp / np.sum(exp, keepdims=True)
        
        
    def loss(self, output, labels):
        """
        Loss function: cross-entropy = -sum_c y * log(p)
        """
        return - np.sum(labels * np.log(output))       

    
    def backward(self, epsilon, labels):
        """
        Back Propagation: Folllws algo 6.4 from DL book p.206
        """
        # Softmax output
        y = self.layers[-1].feature

        # derivative of the loss with respect to softmax output
        grad = - labels / y

        # Now with relu
        for i in range(self.n_layers, 1, -1):
            
            grad = grad * self.grad_act_fct(layer[i])
            
            # with respec to weights
            grad_w = grad * self.layers[i-1].feature
            
            # Update weights
            self.layer[i].layer_weights += - epsilon * grad_w
            
            # Update grad
            grad = self.layers[i-1].layer_weights.T @ grad
               
    
    def update(self):
        """ 
        Parameters update 
        """
        raise NotImplementedError('Method not used.')
        
        
    def train(self, data, target, batch_size=100, learning_rate=1e-1, n_epochs=10):
        """ 
        Train
        """        
        loss_tracking = list()
        for i in range(n_epochs):
        
            epoch_loss = list()
            for i in range(data.shape[0] // batch_size):
                xi = data[i * batch_size : (i + 1) * batch_size]
                yi = target[i * batch_size : (i + 1) * batch_size]
                
                # Forward pass
                loss = self.forward(xi, yi)
                epoch_loss.append(loss)
                
                # Backward pass with weights update
                self.backward()
        
        # Average eopch loss
        loss_tracking.append(sum(epoch_loss) / len(epoch_loss))
        
        return loss_tracking, self.layers

        
    def test(self):
        """
        Test algorithm
        """
        
        
    def grad_act_fct(self, layer):
        
        if layer.activation_function == 'relu':
            return self.grad_relu(layer.activation)
        
        elif layer.activation_function == 'softmax':
            return self.softmax_grad(layer.activation)
  
    
    def grad_relu(self, x):
        """
        Relu derivative
        """
        x[x<=0] = 0
        x[x>0] = 1
        return x
 
    
    def grad_softmax(self, x):
        """
        Softmax derivative
        """        
        x[x<=0] = 0
        x[x>0] = 1
        return x    

           
if __name__ == "__main__":
    

    h0 = 784
    h1 = 512
    h2 = 256
    h3 = 10
    n_params = (h0 + 1) * h1 + (h1 + 1) * h2 + (h2 + 1) * h3
    print("The number of parameters is: {}".format(n_params))