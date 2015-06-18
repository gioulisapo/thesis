import csv
import network
import time
import mnist_loader



#Setup Network
Hidden_Layers=[10]

#Add input.output layers
Layers=[784]+Hidden_Layers+[10]
init='default' #default | large | all_zero
neurons='tanh' #sigmoid | tanh
#Setup Learning Algorithm
epochs=4
Optimisation_technique='L2' #L1 | L2 | none
lmbda=10
Mini_Batch_Size=20;
Learning_Rate=0.25;
Cost_Function=network.CrossEntropyCost() #network2.CrossEntropyCost() | network2.QuadraticCost()

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
    
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network.Network(Layers, cost=Cost_Function, initialization=init, neuron_type=neurons)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, Optimisation_technique, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=False, monitor_training_accuracy=True, monitor_training_cost=False)