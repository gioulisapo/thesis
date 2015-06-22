# import conv
import csv
import network
from network import sigmoid, tanh, ReLU, Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
import time



def mlp(n=1, epochs=5, mini_batch_size=20,lmbda=0.1, Learning_Rate=5):
    nets = []
    for j in range(n):
        print "A deep net with 300, 100 hidden neurons"
        net = Network([
            FullyConnectedLayer(n_in=784, n_out=100, activation_fn=sigmoid, p_dropout=0.0), #784 | 3072, x, 
            # FullyConnectedLayer(n_in=3072, n_out=500),
            # FullyConnectedLayer(n_in=1000, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, Learning_Rate, validation_data, test_data, lmbda)
        nets.append(net)
    return nets


def basic_CNN(n=3, epochs=5, mini_batch_size=20,lmbda=0.1, Learning_Rate=5):
    for j in range(n):
        print "Conv + FC architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=tanh), #sigmoid || tanh || ReLU
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
    return net 

#start---------------------Choices---------------------start
#Choose dataset
dataset='minst' #minst | cifar_10
#Setup Network Hidden Layers

#Choose weight and bias function
# init='default' #default | large | all_zero
# #Choose neuron type
# neurons='sigmoid' #sigmoid | tanh
# #Number of training epochs
epochs=4
# Optimisation_technique='L2'
lmbda=10
# #Choose size of mini batch 1 for inline
Mini_Batch_Size=10;
#Choose Learning rate of gradient descent
Learning_Rate=0.45;
# #Choose cost function
# Cost_Function=network.CrossEntropyCost() #network2.CrossEntropyCost() | network2.QuadraticCost()
# #end---------------------Choices---------------------end

# #Add input.output layers

if dataset=='cifar_10':
    training_data, validation_data, test_data = network.load_data_cifar()
else:
    training_data, validation_data, test_data = network.load_data_mnist()
# print  network.size(training_data)
# training_x, training_y = training_data
# print training_x
# print training_y
# print training_x.shape
# print training_y.shape  

# net=mlp(1,epochs,Mini_Batch_Size,lmbda,Learning_Rate) #(n=1, epochs=5, mini_batch_size=20,lmbda=0.1, Learning_Rate=5)
basic_CNN(1,epochs,Mini_Batch_Size,lmbda,Learning_Rate)
#conv.ensemble(net)
