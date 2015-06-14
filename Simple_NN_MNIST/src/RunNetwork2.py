from __future__ import print_function
import network2
import time
import mnist_loader
resultF = open("tests.md", "a")

#Setup Network
Hidden_Layers=[80,300,80]
#Hidden_Layers=[10]
init='default' #default | large | all_zero
neurons='tanh' #sigmoid | tanh
#Setup Learning Algorithm
epochs=30
Optimisation_technique='none' #L1 | L2 | none
lmbda=0
Mini_Batch_Size=20;
Learning_Rate=0.25;
Cost_Function=network2.CrossEntropyCost() #network2.CrossEntropyCost() | network2.QuadraticCost()

#Prints
print ('# Test', file = resultF)
print ('* The network consist of:',len(Hidden_Layers) ,'Hidden Layers', file = resultF)
for i in range(0,len(Hidden_Layers)):
	print ('\t* Hidden Layer', i+1,'consist of:',Hidden_Layers[i],'hidden',neurons,'neurons', file = resultF)
Layers=[784]+Hidden_Layers+[10]
print ('* The network will run for:',epochs ,'epochs', file = resultF)
print ('* Learning Algorithm: Stochastic Gradient Descent with Mini batch size of:',Mini_Batch_Size,\
	'and Learning Rate:',Learning_Rate, file = resultF)
if Optimisation_technique=='L2':
	print ('* Optimisation Technique: L2 with lmbda=',lmbda, file = resultF)
elif Optimisation_technique=='L1':
	print ('* Optimisation Technique: L1 with lmbda=',lmbda, file = resultF)
else:
	print ('* Optimisation Technique: none', file = resultF)
print ('* Initialization method: ', file = resultF)
if init=='default':
	print ('\t* Weights: Random, (Gaussian with mean=0 and variance=1)/sqrt(Number_Of_Neuron_Connection_Inputs)', file = resultF)
	print ('\t* Biases: Random, Gaussian with mean=0 and variance=1', file = resultF)
elif init=='all_zero':
	print ('\t* Weights: All to 0', file = resultF)
	print ('\t* Biases: All to 0', file = resultF)
else:
	print ('\t* Weights: Random, Gaussian with mean=0 and variance=1', file = resultF)
	print ('\t* Biases: Random, Gaussian with mean=0 and variance=1', file = resultF)
if Cost_Function.__class__.__name__=='CrossEntropyCost':
	print ('* Cost Function: Cross Entropy', file = resultF)
else:
	print ('* Cost Function: Quadratic Cost', file = resultF)
print ('\n---------------------------------------------------------', file = resultF)

resultF.close()
#Train
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network2.Network(Layers, cost=Cost_Function, initialization=init, neuron_type=neurons)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, Optimisation_technique, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)




#Setup Network
Hidden_Layers=[80,300,80]
#Hidden_Layers=[10]
init='default' #default | large | all_zero
neurons='sigmoid' #sigmoid | tanh
#Setup Learning Algorithm
epochs=30
Optimisation_technique='none' #L1 | L2 | none
lmbda=0
Mini_Batch_Size=20;
Learning_Rate=0.25;
Cost_Function=network2.CrossEntropyCost() #network2.CrossEntropyCost() | network2.QuadraticCost()

#Prints
print ('# Test', file = resultF)
print ('* The network consist of:',len(Hidden_Layers) ,'Hidden Layers', file = resultF)
for i in range(0,len(Hidden_Layers)):
	print ('\t* Hidden Layer', i+1,'consist of:',Hidden_Layers[i],'hidden',neurons,'neurons', file = resultF)
Layers=[784]+Hidden_Layers+[10]
print ('* The network will run for:',epochs ,'epochs', file = resultF)
print ('* Learning Algorithm: Stochastic Gradient Descent with Mini batch size of:',Mini_Batch_Size,\
	'and Learning Rate:',Learning_Rate, file = resultF)
if Optimisation_technique=='L2':
	print ('* Optimisation Technique: L2 with lmbda=',lmbda, file = resultF)
elif Optimisation_technique=='L1':
	print ('* Optimisation Technique: L1 with lmbda=',lmbda, file = resultF)
else:
	print ('* Optimisation Technique: none', file = resultF)
print ('* Initialization method: ', file = resultF)
if init=='default':
	print ('\t* Weights: Random, (Gaussian with mean=0 and variance=1)/sqrt(Number_Of_Neuron_Connection_Inputs)', file = resultF)
	print ('\t* Biases: Random, Gaussian with mean=0 and variance=1', file = resultF)
elif init=='all_zero':
	print ('\t* Weights: All to 0', file = resultF)
	print ('\t* Biases: All to 0', file = resultF)
else:
	print ('\t* Weights: Random, Gaussian with mean=0 and variance=1', file = resultF)
	print ('\t* Biases: Random, Gaussian with mean=0 and variance=1', file = resultF)
if Cost_Function.__class__.__name__=='CrossEntropyCost':
	print ('* Cost Function: Cross Entropy', file = resultF)
else:
	print ('* Cost Function: Quadratic Cost', file = resultF)
print ('\n---------------------------------------------------------', file = resultF)

resultF.close()
#Train
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network2.Network(Layers, cost=Cost_Function, initialization=init, neuron_type=neurons)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, Optimisation_technique, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
