# Datasets
The following datasets are available in every NN implementation in this Project

## MNIST Dataset

The [MNIST data set](http://yann.lecun.com/exdb/mnist/) (Mixed National Institute of Standards and Technology database) of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centred in a fixed-size image.

The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centred in a 28x28 image by computing the centre of mass of the pixels, and translating the image so as to position this point at the centre of the 28x28 field. 

### mnist.pkl.gz
As in most Python implementations, the dataset is not being used in its original form. Instead it has been packaged using a Python object serialization library called [Pickle](https://docs.python.org/2/library/pickle.html). After being pickled the dataset is then compressed using the gzip format.
The pickled file [file](http://deeplearning.net/data/mnist/mnist.pkl.gz) represents a tuple of 3 lists:

* Training Set: Containing 50000 tuples
* Validation Set: Containing 10000 tuples
* Test Set: Containing 10000 tuples

Each of the three lists is a pair formed from a list of images **x** and a list of class labels for each of the images **y** forming a 2-d tuple (x,y) where:

* **x:** List of Images: An image is represented as [numpy](http://www.numpy.org/) 1-dimensional array of 784 (28 x 28) float values between 0 and 1 (0 stands for black, 1 for white).
* **y:** List of Labels: The labels are numbers between 0 and 9 indicating which digit the image represents.

The code block below shows how to load the dataset:

```python
import cPickle, gzip, numpy
load():
# Load the dataset
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
```
The code block below shows how to use the aforementioned load() function:
```python
tr_d, va_d, te_d = load_data()
```

## CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

There are 3 versions available for download in the [official Cifar-10 website](http://www.cs.toronto.edu/~kriz/cifar.html)
* [Python Version](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* [Matlab Version](http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz)
* [Binary Version](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

### Python/Matlab Version
The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with cPickle. Here is a Python routine which will open such a file and return a dictionary: 
```python
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict)
```
Loaded in this way, each of the batch files contains a dictionary with the following elements:
* **data** -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the  green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
* **labels** -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:

* label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc

The labels are the following:
* airplane 
* automobile 
* bird 
* cat 
* deer 
* dog 
* frog 
* horse 
* ship 
* truck