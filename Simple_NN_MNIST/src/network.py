"""network.py
~~~~~~~~~~~~~~

Learning algorithm  :Stochastic gradient descentfor a feedforward neural network.
                     Gradients are calculated using backpropagation
Neurons             :Sigmoid | tanh
Loss functions      :Quadradic Loss function | Cross-Entropy
W initialization    :Random, (Gaussian mean=0, variance=1)/sqrt(Number_Of_Neuron_Connection_Inputs) | Same as network1
b initialization    :Random, Gaussian mean=0, variance=1
Optimisation        :L2(Weight Decay) | none

added features      :Added choice between initialization tequniques (class argument)
                    :Added all_zero_initializer weight,bias init tequnique
                    :Added L1 optimisation + no optimisation
                    :Added choice between optimisation tequniques (SGD argument)
                    :Added Print Results in file
                    :Added Print in cvs format
                    :Added Training Process loader
                    :Added neuron choice + tanh neurons

known issues        :L1 optimisation
"""

#### Libraries
# Standard library
import json
import random
import sys
import time
import csv
import math
# Third-party libraries
import numpy as np
#np.seterr(all='ignore')
neurons='' #sigmoid | tanh
#### Define the quadratic and cross-entropy cost functions

class QuadraticCost:

#########################################################################
    pass

#########################################################################
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        if neurons=='sigmoid':
            return (a-y) * sigmoid_prime_vec(z)
        else:
            return (a-y) * tanh_prime_vec(z)

class CrossEntropyCost:

#########################################################################
    pass

#########################################################################
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network():

    def __init__(self, sizes, cost=CrossEntropyCost, initialization='default', neuron_type='sigmoid'):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        global neurons
        neurons=neuron_type
#########################################################################
        if initialization=='default':
            self.default_weight_initializer()
        elif initialization=='all_zero':
            self.all_zero_initializer()
        else:
            self.large_weight_initializer()
#########################################################################
        self.cost=cost

    def default_weight_initializer(self): #New Method!!!
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

#########################################################################
    def all_zero_initializer(self): #Same as before
        """Initialize the weights as well as the biases to zeros 
        """
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.weights = [np.zeros((y, x)) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
#########################################################################

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        if(neurons=='sigmoid'):
            for b, w in zip(self.biases, self.weights):
                a = sigmoid_vec(np.dot(w, a)+b)
        else:
            for b, w in zip(self.biases, self.weights):
                a = tanh_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            optimisation = 'L2',
            lmbda = 0.0, 
            evaluation_data=None, 
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False, 
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists``: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
#########################################################################
        sys.stdout.write('\r')
        CpuTrainingTime=0.0
        sys.stdout.write("Training Process: [%-20s] %d%%" % ('='*0, 0))
        sys.stdout.flush()

        resultF = open('Results.csv', 'a')
        writer = csv.writer(resultF)
        writer.writerow(['Epoch','Training Data Cost','Traning Data Accuracy','Evaluation Data Cost','Evaluation Data Accuracy','Time'])
#########################################################################
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            start=time.clock() #Any extra calculations will not be calculated as training time
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, optimisation, len(training_data))
#########################################################################
            CpuTrainingTime+=time.clock()-start #Any extra calculations will not be calculated as training time
            tcost=0.0
            taccuracy=0.0
            ecost=0.0
            eaccuracy=0.0
            if monitor_training_cost:
                tcost = self.total_cost(training_data, lmbda, optimisation)
                training_cost.append(tcost)
            if monitor_training_accuracy:
                taccuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(taccuracy)
            if monitor_evaluation_cost:
                ecost = self.total_cost(evaluation_data, lmbda, optimisation, convert=True)
                evaluation_cost.append(ecost)
            if monitor_evaluation_accuracy:
                eaccuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(eaccuracy)
            try:
                writer.writerow([j+1,tcost,(taccuracy/float(n)),ecost,(eaccuracy/float(n_data)),CpuTrainingTime])
            except IOError:
                print 'cannot write'
            sys.stdout.write('\r')
            sys.stdout.write("Training Process: [%-20s] %d%%" % ('='*int(float(j+1)/epochs*20), (float(j+1)/epochs*100)))
            sys.stdout.flush()
            
#########################################################################
            # if monitor_training_cost:
            #     cost = self.total_cost(training_data, lmbda, optimisation)
            #     training_cost.append(cost)
            #     print '_Cost on training data: {}_\t\t'.format(cost)
            # if monitor_training_accuracy:
            #     accuracy = self.accuracy(training_data, convert=True)
            #     training_accuracy.append(accuracy)
            #     print '_Accuracy on training data: {} / {}_\t'.format(accuracy, n)
            # if monitor_evaluation_cost:
            #     cost = self.total_cost(evaluation_data, lmbda, optimisation, convert=True)
            #     evaluation_cost.append(cost)
            #     print "_Cost on evaluation data: {}_\t".format(cost)
            # if monitor_evaluation_accuracy:
            #     accuracy = self.accuracy(evaluation_data)
            #     evaluation_accuracy.append(accuracy)
            #     print "_Accuracy on evaluation data: {} / {}_\t\t".format(self.accuracy(evaluation_data), n_data)
#########################################################################
        writer.writerow(['----------------','-------------------','--------------------','-------------------','---------------------','-----------------'])
        writer.writerow(['----------------','-------------------','--------------------','-------------------','---------------------','-----------------'])
        resultF.close()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, optimisation, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#########################################################################
        if optimisation=='L2':
            self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
        elif optimisation=='L1':
            self.weights = [(1-eta*(lmbda/n))*np.abs(w)-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
        else: #none
            self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
#########################################################################
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        if neurons=='sigmoid':
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid_vec(z)
                activations.append(activation)
        else:
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = tanh_vec(z)
                activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        if neurons=='sigmoid':
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = sigmoid_prime_vec(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        else:
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = tanh_prime_vec(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.  

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) 
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, optimisation, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
#########################################################################
        if optimisation=='L2':
            cost += 0.5*(lmbda/len(data))*sum(
                np.linalg.norm(w)**2 for w in self.weights)
        elif optimisation=='L1':
            cost += 0.5*(lmbda/len(data))*sum(
                abs(np.linalg.norm(w)) for w in self.weights)
        else:
            cost += 0.5*(lmbda/len(data))
#########################################################################
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#########################################################################
def tanh(z):
    """The tanh sigmoid function """
    return np.tanh(z)
tanh_vec = np.vectorize(tanh)

def tanh_prime(z):
    """Derivative of the tanh sigmoid function."""
    return 1.0-(tanh(z)*tanh(z))

tanh_prime_vec = np.vectorize(tanh_prime)

#########################################################################
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)
