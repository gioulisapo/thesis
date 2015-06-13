__author__ = "Apostolis Gioulis"

import mnist_loader
import network2
import timeit



#Setup Network
Hidden_Layers=[100,350,100]
init='all_zero' #default | large | all_zero
#Setup Learning Algorithm
epochs=60
lmbda=0
Mini_Batch_Size=5;
Learning_Rate=0.25;
Cost_Function=network2.QuadraticCost() #network2.CrossEntropyCost() | network2.QuadraticCost()

#Prints
print '# Test'
print '* The network consist of:',len(Hidden_Layers) ,'Hidden Layers'
for i in range(0,len(Hidden_Layers)):
	print '\t* Hidden Layer', i+1,'consist of:',Hidden_Layers[i],'hidden units'
Layers=[784]+Hidden_Layers+[10]
print '* The network will run for:',epochs ,'epochs'
print '* Learning Algorithm: Stochastic Gradient Descent with Mini batch size of:',Mini_Batch_Size, 'and Learning Rate:',Learning_Rate
if lmbda!=0:
	print '* Optimisation Technique: L2 with lmbda=',lmbda
else:
	print '* Optimisation Technique: none'
print '* Initialization method: '
if init=='default':
	print '\t* Weights: Random, (Gaussian with mean=0 and variance=1)/sqrt(Number_Of_Neuron_Connection_Inputs)'
	print '\t* Biases: Random, Gaussian with mean=0 and variance=1'
elif init=='all_zero':
	'\t* Weights: All to 0'
	'\t* Biases: All to 0'
else:
	print '\t* Weights: Random, Gaussian with mean=0 and variance=1'
	print '\t* Biases: Random, Gaussian with mean=0 and variance=1'
if Cost_Function.__class__.__name__=='CrossEntropyCost':
	print '* Cost Function: Cross Entropy'
else:
	print '* Cost Function: Quadratic Cost'
print '\n---------------------------------------------------------'
#Train
start = timeit.default_timer()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network2.Network(Layers, cost=Cost_Function, initialization=init)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
stop = timeit.default_timer()

#Print Results
seconds=stop - start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print "### Time ran: %d:%02d:%02d" % (h, m, s)
print '\n---------------------------------------------------------'
print '---------------------------------------------------------'


#Setup Network
Hidden_Layers=[100,350,100]
init='large' #default | large | all_zero
#Setup Learning Algorithm
epochs=60
lmbda=0
Mini_Batch_Size=5;
Learning_Rate=0.25;
Cost_Function=network2.QuadraticCost() #network2.CrossEntropyCost() | network2.QuadraticCost()

#Prints
print '# Test'
print '* The network consist of:',len(Hidden_Layers) ,'Hidden Layers'
for i in range(0,len(Hidden_Layers)):
	print '\t* Hidden Layer', i+1,'consist of:',Hidden_Layers[i],'hidden units'
Layers=[784]+Hidden_Layers+[10]
print '* The network will run for:',epochs ,'epochs'
print '* Learning Algorithm: Stochastic Gradient Descent with Mini batch size of:',Mini_Batch_Size, 'and Learning Rate:',Learning_Rate
if lmbda!=0:
	print '* Optimisation Technique: L2 with lmbda=',lmbda
else:
	print '* Optimisation Technique: none'
print '* Initialization method: '
if init=='default':
	print '\t* Weights: Random, (Gaussian with mean=0 and variance=1)/sqrt(Number_Of_Neuron_Connection_Inputs)'
	print '\t* Biases: Random, Gaussian with mean=0 and variance=1'
elif init=='all_zero':
	'\t* Weights: All to 0'
	'\t* Biases: All to 0'
else:
	print '\t* Weights: Random, Gaussian with mean=0 and variance=1'
	print '\t* Biases: Random, Gaussian with mean=0 and variance=1'
if Cost_Function.__class__.__name__=='CrossEntropyCost':
	print '* Cost Function: Cross Entropy'
else:
	print '* Cost Function: Quadratic Cost'
print '\n---------------------------------------------------------'
#Train
start = timeit.default_timer()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network2.Network(Layers, cost=Cost_Function, initialization=init)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
stop = timeit.default_timer()

#Print Results
seconds=stop - start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print "### Time ran: %d:%02d:%02d" % (h, m, s)
print '\n---------------------------------------------------------'
print '---------------------------------------------------------'



#Setup Network
Hidden_Layers=[100,350,100]
init='default' #default | large | all_zero
#Setup Learning Algorithm
epochs=60
lmbda=0
Mini_Batch_Size=5;
Learning_Rate=0.25;
Cost_Function=network2.QuadraticCost() #network2.CrossEntropyCost() | network2.QuadraticCost()

#Prints
print '# Test'
print '* The network consist of:',len(Hidden_Layers) ,'Hidden Layers'
for i in range(0,len(Hidden_Layers)):
	print '\t* Hidden Layer', i+1,'consist of:',Hidden_Layers[i],'hidden units'
Layers=[784]+Hidden_Layers+[10]
print '* The network will run for:',epochs ,'epochs'
print '* Learning Algorithm: Stochastic Gradient Descent with Mini batch size of:',Mini_Batch_Size, 'and Learning Rate:',Learning_Rate
if lmbda!=0:
	print '* Optimisation Technique: L2 with lmbda=',lmbda
else:
	print '* Optimisation Technique: none'
print '* Initialization method: '
if init=='default':
	print '\t* Weights: Random, (Gaussian with mean=0 and variance=1)/sqrt(Number_Of_Neuron_Connection_Inputs)'
	print '\t* Biases: Random, Gaussian with mean=0 and variance=1'
elif init=='all_zero':
	'\t* Weights: All to 0'
	'\t* Biases: All to 0'
else:
	print '\t* Weights: Random, Gaussian with mean=0 and variance=1'
	print '\t* Biases: Random, Gaussian with mean=0 and variance=1'
if Cost_Function.__class__.__name__=='CrossEntropyCost':
	print '* Cost Function: Cross Entropy'
else:
	print '* Cost Function: Quadratic Cost'
print '\n---------------------------------------------------------'
#Train
start = timeit.default_timer()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network2.Network(Layers, cost=Cost_Function, initialization=init)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )
net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, lmbda, evaluation_data=validation_data,\
	monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
stop = timeit.default_timer()

#Print Results
seconds=stop - start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print "### Time ran: %d:%02d:%02d" % (h, m, s)
print '\n---------------------------------------------------------'
print '---------------------------------------------------------'

 