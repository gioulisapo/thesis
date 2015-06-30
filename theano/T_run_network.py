#!/usr/bin/env python
'''T_run_network.py
~~~~~~~~~~~~~~
    This module is used to build and train neural Neworks using a numpy NN implementation ./network
    based on the work of Michael Nielsen (https://github.com/mnielsen/neural-networks-and-deep-learning)
    using ./cifar_loader.py | ./mnist_loader.py files to load the appropriate datset
    The following examples are a small sample form the /conv.py file by Nielsen.
'''
__author__ = "Apostolos Gioulis"

#Imports
import csv
import configparser
import network
from T_network import sigmoid, tanh, ReLU, Network
from T_network import FullyConnectedLayer, SoftmaxLayer
import time


def mlp(training_data ,validation_data ,test_data, Input_Neurons, Neurons=sigmoid, n=1, epochs=5, mini_batch_size=20,lmbda=0.1, Learning_Rate=5):
    '''
    This Function based on the shallow() experiment in the conv.py module by Michael Nielsen (https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/conv.py)
    This Function creates and trains a Multilayered Perceptron of custom Depth. Some of the training options
    are given to the user as function arguments.

    Layer combinations must have the two following characteristics:

        - Input Layer must have the same number of neurons as the given input vis-a-vi
            - When using MNISt dataset: Input neurons must amount to 784
            - When using Cifar-10 dataset Input neurons must amount to 3072
        - Each Fully Connected Layer must have the same number of inputs as the outputs of the previous Layer
        - Output Layers must be Maxout

    Usage Example

        mlp(training_data, validation_data, test_data, 1,epochs, Mini_Batch_Size, lmbda, Learning_Rate)

    Known Issues:

        Cifar-10 networks don't seem to improve their performance
    '''
    if Input_Neurons==784:
        dataset="MNIST"
    else:
        dataset="Cifar-10"
    if Neurons==sigmoid:
        N_Type="sigmoid"
    elif Neurons==ReLU:
        N_Type="ReLu"
    else:
        N_Type="tanh"

    for j in range(n):
        print ("A deep net with 300, 100 {0} hidden neurons training for {1} epochs in the {2} dataset".format(N_Type, epochs, dataset))
        print ("Using a Mini Batch Size of {0} inputs, a Learning Rate of {1} and Regularization Param of {2}".format(mini_batch_size, Learning_Rate, lmbda))
        print '----------------------------------------------------------------------------------------------------------------'
        net = Network([
            FullyConnectedLayer(n_in=Input_Neurons, n_out=300,activation_fn=Neurons),
            FullyConnectedLayer(n_in=300, n_out=100, activation_fn=Neurons),
            SoftmaxLayer(n_in=100, n_out=10)],mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, Learning_Rate, validation_data, test_data, lmbda)
    return

def main():
    '''
    This Function starts of, by loading configuration parameters (./network.ini), used to build and train the DNN.
    After loading configuration parameters the proper dataset is loaded and a DNN is created and then trained using SGD
    calling the constructor of the Network class and it's SGD method accordingly.
    The Layers of the DNN are predetermined but can be tampered with by editing the mlp() function.
    '''
    config = configparser.ConfigParser() #Load Config Parser
    config.read('T_network.ini') #Read Config File

    dataset=config['Network']['dataset'] #Start Loading network parameters
    Neurons=config['Network']['Neurons'] #Start Loading network parameters

    epochs = int(config['Learning-Training']['Epochs'])  #Start Loading Learning-Training process parameters
    lmbda = float(config['Learning-Training']['Lmbda'])
    Mini_Batch_Size = int(config['Learning-Training']['Mini_Batch_Size'])
    Learning_Rate = float(config['Learning-Training']['Learning_Rate'])

    if dataset=='cifar_10':
        training_data, validation_data, test_data = network.load_data_cifar()
        Input_Neurons=3072
    else: #dataset
        training_data, validation_data, test_data = network.load_data_mnist()
        Input_Neurons=784
    if Neurons=='sigmoid':
        Neurons=sigmoid
    elif Neurons=='tanh':
        Neurons=tanh
    else :
        Neurons=ReLU

    mlp(training_data ,validation_data ,test_data, Input_Neurons, Neurons, 1, epochs, Mini_Batch_Size,lmbda, Learning_Rate)

if __name__ == "__main__":
    main()
