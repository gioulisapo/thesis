#!/usr/bin/env python
'''run_network.py
~~~~~~~~~~~~~~
    This module is used to build and train neural Neworks using a numpy NN implementation ./network
    based on the work of Michael Nielsen (https://github.com/mnielsen/neural-networks-and-deep-learning)
    using ./cifar_loader.py | ./mnist_loader.py files to load the appropriate datset

'''
__author__ = "Apostolos Gioulis"

#Imports
import csv
import network
import configparser

#Main Method
def main():
    '''
        This Function starts of, by loading configuration parameters (./network.ini), used to build and train the NN.
        After loading configuration parameters the proper dataset is loaded and a NN is created and then trained using SGD
        calling the constructor of the Network class and it's SGD method accordingly.
        All results will be appended in ./Results.csv file in the following format:
            Network/Training settings
            ----------------------------------------------------------------------------------------------------------------
            Results
            ...
            ...
            ----------------------------------------------------------------------------------------------------------------
            ----------------------------------------------------------------------------------------------------------------
    '''
    config = configparser.ConfigParser() #Load Config Parser
    config.read('network.ini') #Read Config File

    dataset=config['Network']['Dataset'] #Start Loading network parameters
    Hidden_Layers=config['Network']['Hidden_Layers']
    Hidden_Layers = map(int, Hidden_Layers.split(','))
    init = config['Network']['Init_Funct']
    if config['Network']['Cost_Function'] == 'QuadraticCost':
        Cost_Function=network.QuadraticCost()
    else:
        Cost_Function=network2.QuadraticCost()
    neurons = config['Network']['Neurons']

    epochs = int(config['Learning-Training']['Epochs'])  #Start Loading Learning-Training process parameters
    Regularization_technique = config['Learning-Training']['Regularization_technique']
    lmbda = float(config['Learning-Training']['Lmbda'])
    Mini_Batch_Size = int(config['Learning-Training']['Mini_Batch_Size'])
    Learning_Rate = float(config['Learning-Training']['Learning_Rate'])

    # Add input,output layers to the NN-DNN according to the chosen dataset
    if dataset=='cifar_10':
        import cifar_loader
        Layers=[3072]+Hidden_Layers+[10]
        training_data, validation_data, test_data = cifar_loader.load_data()
    else:
        import mnist_loader
        Layers=[784]+Hidden_Layers+[10]
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Write Chosen Configuration for the upcoming experiment
    resultF = open('Results.csv', 'at')
    try:
        writer = csv.writer(resultF)
        writer.writerow(['Network:', Layers,'Neurons:', neurons])
        writer.writerow(['Using Dataset:', dataset])
        writer.writerow(['Epochs:', epochs,'Mini_Batch_Size:', Mini_Batch_Size])
        writer.writerow(['Init:', init,'Learning_Rate:', Learning_Rate])
        writer.writerow(['Regularization:', Regularization_technique,lmbda])
        writer.writerow(['Cost Function:',  Cost_Function.__class__.__name__])
        writer.writerow(['----------------','-------------------','--------------------','-------------------','---------------------','-----------------'])
    finally:
    	resultF	.close()

    # Build Network
    net = network.Network(Layers, cost=Cost_Function, initialization=init, neuron_type=neurons)
    # Start Training
    net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, Regularization_technique, lmbda, evaluation_data=validation_data,\
     	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

if __name__ == "__main__":
    main()
