# Deep Learning Thesis Project
## by: Apostolis Gioulis

The projects purpose is to create a work-flow using some of the available tools used in modern Deep Learning algorithm Development. To be specific, Theano and Torch7 libraries are used as well as a custom python implementation using Python.

## Implementations
There are 3 folders containing 3 different implementations as well as some other modules of the supportive nature:
* __numpy/:__ Contains a NN implementation using python extended with the scientific computing library [numpy](http://www.numpy.org/). The implementation essentially is an extension of the [code](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py) published by Michael Nielsen for education purposes. All is downloaded in the data/ folder located at the same level as the thesis root folder.
* __theano/:__ Contains a NN implementation written in Python using the [theano](http://deeplearning.net/software/theano/) framework. The implementation is also based on the [work](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py) of M.Nielsen
* __Torch7:__ Contains a NN implementation using [Torch7](http://torch.ch/) framework as well as the [dp library](https://github.com/nicholas-leonard/dp) by Nicholas Leonard
The code used is actually an [example](https://github.com/nicholas-leonard/dp/blob/master/examples/deepinception.lua) provided by Nicholas Leonard

## Requirements
* __numpy/:__
In order for the numpy implementation to run successfully the following python packages must be installed.
	* numpy
	* configparser
	* clint
	* requests

	They can be installed by using the ```sudo pip install <package_name>``` command


* __theano/:__ In order for the theano implementation to run successfully:
theano library must be installed. To do so either:
	* Use the automated tool ```theano/theano_setup.sh``` developed for the purposes of the work-flow (tested in fedora21, ubuntu 14.04)
	* Follow the instructions available in the official theano [website](http://deeplearning.net/software/theano/install.html)

	All of the aforementioned python packages are required as well.
* __Torch7:__
In order for the torch7 NN implementation to run torch7 must be installed in the users system. To do so:
	* Use the automated tool ```Torch7/torch7_setup.sh``` that in turn uses the [official installation tool](https://github.com/torch/ezinstall). If the script does not work some debugging options are listed as comments in the script. The script in turn requires
		* cmake
		* curl

	Once torch7 is installed the dp library must be installed as well (```Torch7/torch7_setup.sh``` installs it automatically by executing)
	```bash
	luarocks install dp --local
	```
## Execution

* __numpy/:__
	* Configure: By tampering with the  numoy/src/network.ini file
	* Run: ```./run_network.py```
* __theano/:__
	* Configure: By tampering with the  theano/src/network.ini file
	* Run: ```./T_run_network.py```
* __Torch7:__:
	* Configure: There are many Configuration options by passing command line args. For more info on the available options read line 16-32 of the file
	* Example Run: ```th simple_NN.lua --learningRate 0.25 --dataset Mnist```
