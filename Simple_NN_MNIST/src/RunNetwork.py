__author__ = "Apostolis Gioulis"

import mnist_loader
import network
import timeit


print '# Test'
Hidden_Layers=[30];

print '* The network consist of:',len(Hidden_Layers) ,'Hidden Layers'
for i in range(0,len(Hidden_Layers)):
	print '\t* Hidden Layer', i+1,'consist of:',Hidden_Layers[i],'hidden units'
Layers=[784]+Hidden_Layers+[10]
epochs=3
Mini_Batch_Size=5;
Learning_Rate=0.25;
print '\n* The network will run for:',epochs ,'epochs'
print '* Learning Algorithm: Stochastic Gradient Descent with Mini batch size of:',Mini_Batch_Size, 'and Learning Rate:',Learning_Rate

start = timeit.default_timer()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Called Network[Sizeof(input layer), Sizeof(hidden_layer_1), ... , Sizeof(Output_Layer)]
net = network.Network(Layers)
#SGD(Taining_Data, epochs, Mini_Batch_Size, Learning_Rate,test_data=Test_Data_For_Evaluation )

net.SGD(training_data, epochs, Mini_Batch_Size, Learning_Rate, test_data=test_data)
stop = timeit.default_timer()
print '\n---------------------------------------------------------'
seconds=stop - start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print "### Time ran: %d:%02d:%02d" % (h, m, s)
print '\n---------------------------------------------------------'
print '---------------------------------------------------------'
