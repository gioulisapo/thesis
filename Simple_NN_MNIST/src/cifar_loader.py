


import cPickle
import os.path
import numpy as np

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
	f=unpickle('../../../data/cifar-10-batches-py/data_batch_1')
	v=unpickle('../../../data/cifar-10-batches-py/data_batch_2')
	t=unpickle('../../../data/cifar-10-batches-py/test_batch')

	# data_tr=f.get('data')
	# labels_tr=f.get('labels')

	# data_val=v.get('data')
	# labels_val=v.get('labels')

	# data_tst=t.get('data')
	# labels_tst=t.get('labels')

	training_data=tuple([np.float32(f.get('data')),f.get('labels')])

	validation_data=tuple([np.float32(v.get('data')),v.get('labels')])

	test_data=tuple([np.float32(t.get('data')),t.get('labels')])

	training_inputs = [np.reshape(x, (3072, 1)) for x in training_data[0]] # returns a list of the images
	training_results = [vectorized_result(y) for y in training_data[1]] # list of numpy.ndarrays (All labels of the training data)
	training_data = zip(training_inputs, training_results) # zip returns a list od tuples that have the following form {(image,label).....}

	validation_inputs = [np.reshape(x, (3072, 1)) for x in validation_data[0]]
	validation_data = zip(validation_inputs, validation_data[1])

	test_inputs = [np.reshape(x, (3072, 1)) for x in test_data[0]]
	test_data = zip(test_inputs, test_data[1])

	return (training_data, validation_data, test_data)

load_data()