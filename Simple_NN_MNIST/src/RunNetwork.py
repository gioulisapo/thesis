import csv
import network
import time
import mnist_loader
import cifar_loader

#start---------------------Choices---------------------start
#Choose dataset
dataset='cifar_10' #minst | cifar_10
#Setup Network Hidden Layers
Hidden_Layers=[300, 400]
#Choose weight and bias function
init='default' #default | large | all_zero
#Choose neuron type
neurons='sigmoid' #sigmoid | tanh
#Number of training epochs
epochs=3
#Choose optimisation technique
Optimisation_technique='L2' #L2 | L2 | none
lmbda=10
#Choose size of mini batch 1 for inline
Mini_Batch_Size=10;
#Choose Learning rate of gradient descent
Learning_Rate=0.25;
#Choose cost function
Cost_Function=network.QuadraticCost() #network2.CrossEntropyCost() | network2.QuadraticCost()
#end---------------------Choices---------------------end

#Add input.output layers
if dataset=='cifar_10':
    Layers=[3072]+Hidden_Layers+[10]
    training_data, validation_data, test_data = cifar_loader.load_data()
else:
    Layers=[784]+Hidden_Layers+[10]
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Prints
resultF = open('Results.csv', 'at')
try:
    writer = csv.writer(resultF)
    writer.writerow(['Network:', Layers,'Neurons:', neurons])
    writer.writerow(['Epochs:', epochs,'Mini_Batch_Size:', Mini_Batch_Size])
    writer.writerow(['Init:', init,'Learning_Rate:', Learning_Rate])
    writer.writerow(['Optimisation:', Optimisation_technique,lmbda])
    writer.writerow(['Cost Function:',  Cost_Function.__class__.__name__])
    writer.writerow(['----------------','-------------------','--------------------','-------------------','---------------------','-----------------'])
finally:
	resultF	.close()
    

print len(training_data)
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network.Network(Layers, cost=Cost_Function, initialization=init, neuron_type=neurons)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, Optimisation_technique, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)