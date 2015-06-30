"""T_network.py
~~~~~~~~~~~~~~
A Theano-based program for training and running simple neural
networks.

Heavily Based on the NN implementation
of Michael Nielsen (https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py)

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Added features
    - Cifar-10 dataset support
    - Show dataset Download Progress
    - Added timing of the training process
Known issues
    - No Visible improvement appears to be made while training NN for the cifar-10 dataset
"""

__author__ = "M. Nielsen, Apostolos Gioulis"

#### Libraries
# Standard library
import cPickle
import gzip
import os.path
# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
#########################################################################
from clint.textui import progress
import requests
import time
#########################################################################


#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."

#### Load the MNIST data

def unpickle(file):
    '''
         unpicles $file and returns the deriving dictionary
    '''
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#########################################################################
def untar(fname, data_path):
    '''
        Calls child process that decompresses $fname to $data_path
        using bash call
    '''
    bashCommand = "tar zxvf "+fname+" -C "+data_path
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

def load_data_cifar(cifar_path="../../data/cifar-10-batches-py"):
    '''
    A method to load the Cifar-10 image data. The method, returns 3 tuples containing cifar-10 data
    In particular, each one of the three results is a list containing 10,000
    2-tuples ``(x, y)``.
        - ``x`` is a 3072-dimensional numpy.ndarray containing the input image.
        - ``y`` is a 10-dimensional numpy.ndarray representing the unit vector corresponding to the correct label for ``x``.
    The cifar-10 data must me inside the $(data_path) folder (data/ folder is the same level as the root folder thesis/)
    If not the data will be downloaded (while displaying download process) and then extracted in order to be loaded
    '''
    cifar_tar = cifar_path.rstrip('cifar-10-batches-py')+'cifar-10-python.tar.gz'
    data_path=cifar_path.rstrip('cifar-10-batches-py')
    if not os.path.exists(cifar_path): # if folder containing the picled data does not exist
        if not os.path.exists(data_path): # create the ../../../data folder if it doesnt exist
            os.makedirs(data_path)
        if os.path.isfile(cifar_tar):
            statinfo = os.stat(cifar_tar)
        if (os.path.isfile(cifar_tar)!=True) or (statinfo.st_size < 170498071): # if the data does not exist in it's tar.gz form download it
            print 'Cifar-10 dataset is missing. Beginning Download ...'  # Start downloading data while displaying dowload process
            r = requests.get('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', stream=True)
            with open(cifar_tar, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()
        untar(cifar_tar, data_path)
        os.remove(cifar_tar)

    f=unpickle('../../data/cifar-10-batches-py/data_batch_1')
    v=unpickle('../../data/cifar-10-batches-py/data_batch_2')
    t=unpickle('../../data/cifar-10-batches-py/test_batch')

    training_data=tuple([np.float32(f.get('data')),np.int32(f.get('labels'))])

    validation_data=tuple([np.float32(v.get('data')),v.get('labels')])

    test_data=tuple([np.float32(t.get('data')),t.get('labels')])

    def shared(data):
        """
        Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]
#########################################################################

def load_data_mnist(filename="../../data/mnist.pkl.gz"):
    '''
    A method to load the MNSIT image data as a tuple containing the training data, the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
        - The first entry contains the actual training images.  This is a numpy ndarray with 50,000 entries.
          Each entry is, in turn, a numpy ndarray with 784 values, representing the 28 * 28 = 784 pixels in a single MNIST image.
        - The second entry in the ``training_data`` tuple is a numpy ndarray containing 50,000 entries.
          Those entries are just the digit values (0...9) for the corresponding images contained in the first entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except each contains only 10,000 images.

    Added features (Not available in the original code od M.Nielsen)

        - Added features (Not available in the original code od M.Nielsen)
        - Added Download dataset option (in case it doesn't already exist)
        - Added Downlad Loader using clint
    '''
#########################################################################
    data_path = filename.rstrip('mnist.pkl.gz')
    if (os.path.isfile(filename)!=True):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        print 'MNIST dataset is missing. Beginning Download'
        r = requests.get('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', stream=True)
        with open(filename, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
#########################################################################
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """
        Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


class Network():
    '''
    Main class used to construct and train networks.
    Usage example:
        net = Network([
        FullyConnectedLayer(n_in=Input_Neurons, n_out=300),
        FullyConnectedLayer(n_in=300, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)],mini_batch_size)
    '''
    def __init__(self, layers, mini_batch_size):
        """
        Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """
        Train the network using mini-batch stochastic gradient descent.

        Usage example:

            net= Network([.......],..)
            net.SGD(training_data, epochs, mini_batch_size, Learning_Rate, validation_data, test_data, lmbda)
        """
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        CpuTrainingTime=0.0
        for epoch in xrange(epochs):
            start=time.clock() #Any extra calculations will not be calculated as training time
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    prev=CpuTrainingTime
                    CpuTrainingTime+=time.clock()-start
                    print("Epoch {0} lasted {1} sec: validation accuracy {2:.2%}".format(
                        epoch, (CpuTrainingTime-prev), validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network in {0} seconds.".format(CpuTrainingTime))
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


# class ConvPoolLayer():
#     """Used to create a combination of a convolutional and a max-pooling
#     layer.  A more sophisticated implementation would separate the
#     two, but for our purposes we'll always use them together, and it
#     simplifies the code, so it makes sense to combine them.

#     """

#     def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
#                  activation_fn=sigmoid):
#         """`filter_shape` is a tuple of length 4, whose entries are the number
#         of filters, the number of input feature maps, the filter height, and the
#         filter width.

#         `image_shape` is a tuple of length 4, whose entries are the
#         mini-batch size, the number of input feature maps, the image
#         height, and the image width.

#         `poolsize` is a tuple of length 2, whose entries are the y and
#         x pooling sizes.

#         """
#         self.filter_shape = filter_shape
#         self.image_shape = image_shape
#         self.poolsize = poolsize
#         self.activation_fn=activation_fn
#         # initialize weights and biases
#         n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
#         self.w = theano.shared(
#             np.asarray(
#                 np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
#                 dtype=theano.config.floatX),
#             borrow=True)
#         self.b = theano.shared(
#             np.asarray(
#                 np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
#                 dtype=theano.config.floatX),
#             borrow=True)
#         self.params = [self.w, self.b]

#     def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
#         self.inpt = inpt.reshape(self.image_shape)
#         conv_out = conv.conv2d(
#             input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
#             image_shape=self.image_shape)
#         pooled_out = downsample.max_pool_2d(
#             input=conv_out, ds=self.poolsize, ignore_border=True)
#         self.output = self.activation_fn(
#             pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#         self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer():
    '''
    Implementation of a Fully Conected Neuron Layer.

    Users can determine:

        - The input connections of the layer as well as the number of neurons the layer consists of $(n_out).
        - Users can also determine the type of neurons (tanh|sigmoid|Relu)
        - Finally a chance of dropout can be added

    The biases are initialized randomly from a normal distribution
    The weights follow the same process with the difference that the random number is then divided by a factor of sqrt(1.0/n_out)

    Usage example

        FullyConnectedLayer(n_in=Input_Neurons, n_out=300,activation_fn=Neurons)
    '''
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        '''
        Initialize layers parameters
            - Weights: Random numbers drawn from a normal distribution divided by a factor of sqrt(1.0/n_out).
            - Biases: Random numbers drawn from a normal distribution.
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer():
    '''
    Implementation of a Fully Conected Neuron Layer.
        - Users can determine the input connections of the layer as well as the number of neurons
    the layer consists of $(n_out).
        - Finally a chace of dropout can be added
    Weights and Biases are initialized to zero

    Usage example:

        FullyConnectedLayer(n_in=Input_Neurons, n_out=300,activation_fn=Neurons)
    '''
    def __init__(self, n_in, n_out, p_dropout=0.0):
        '''
        Initialize layers parameters to zero
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    '''
    Applies dropout to a $(layer) by diactivating a neuron with a pobabilty of $(p_dropout)
    '''
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
