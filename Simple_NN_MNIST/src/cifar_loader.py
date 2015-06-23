from clint.textui import progress
import tarfile
import cPickle
import requests
import os.path
import numpy as np
# import zlib
# import bz2

def unpickle(file):
    '''
        unpicles $file and returns the deriving dictionary
    '''
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

#
def untar(fname, data_path):
    '''
        Calls child process that decompresses $fname to $data_path
        using bash call
    '''
    bashCommand = "tar zxvf "+fname+" -C "+data_path
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

def load_data():
    '''
        Method that returns 3 tuples in the form of X_data={data,labels} containing cifar-10 data
        The cifar-10 data must me inside the $data_path folder
        If not the data will be downloaded and Extracted
    '''
    cifar_path='../../../data/cifar-10-batches-py'
    cifar_tar='../../../data/cifar-10-python.tar.gz'
    data_path='../../../data/'
    if (os.path.isfile(cifar_tar)!=True): #if folder containing the picled data does not exist
        if not os.path.exists(data_path): # create the ../../../data folder if it doesnt exist
            os.makedirs(data_path)
        print 'Cifar-10 dataset is missing. Beginning Download ...'  # Start downloading data while displaying dowload process 
        r = requests.get('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', stream=True)
        with open(cifar_tar, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()
    if(os.path.isfile(cifar_path)!=True):# The tar.gz file exists by now. If needed decompress it
        untar(cifar_tar, data_path)
        os.remove(cifar_tar)
    
    f=unpickle('../../../data/cifar-10-batches-py/data_batch_1')
    v=unpickle('../../../data/cifar-10-batches-py/data_batch_2')
    t=unpickle('../../../data/cifar-10-batches-py/test_batch')
    #Change dormat of data 
    training_data=tuple([np.float32(f.get('data')),f.get('labels')])

    validation_data=tuple([np.float32(v.get('data')),v.get('labels')])

    test_data=tuple([np.float32(t.get('data')),t.get('labels')])
    # As done in mnist_loader.py
    training_inputs = [np.reshape(x, (3072, 1)) for x in training_data[0]] # returns a list of the images
    training_results = [vectorized_result(y) for y in training_data[1]] # list of numpy.ndarrays (All labels of the training data)
    training_data = zip(training_inputs, training_results) # zip returns a list od tuples that have the following form {(image,label).....}

    validation_inputs = [np.reshape(x, (3072, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])

    test_inputs = [np.reshape(x, (3072, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])

    return (training_data, validation_data, test_data)

load_data()