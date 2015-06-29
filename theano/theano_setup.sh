#!/bin/bash
#tested on Fedora_21, Ubuntu_14.04LTS
green='\033[0;32m'
orange='\033[0;33m'
red='\033[0;31m'
NC='\033[0m' # No Color
bold=`tput bold`
normal=`tput sgr0`

#check python version
#----------------------------------------------------------------------------------------------------------------------------------------------#
version="$(python -c 'import sys; print sys.version_info')"
major=`cut -d "=" -f 2 <<< "$version"`
minor=`cut -d "=" -f 3 <<< "$version"`
if [ ${major:0:1} -lt 2 ]; then
	echo -e "${red}Theano requires python.2.6 or more;"
	exit;
elif [[ ${major:0:1} -eq 2 ]]; then
	if [ ${minor:0:1} -ge 6 ]; then
		echo -e "${green}${bold}OK${normal}${NC}\tpython.${major:0:1}.${minor:0:1}"
	else
		echo -e "${red}${bold}Error:${normal}${NC}\tTheano requires python.2.6 or more;"
		exit;
	fi
else
	if [ ${minor:0:1} -le 2 ]; then
		echo -e "${red}${bold}Error:${normal}${NC}\t Python 3 is supported via 2to3 only, starting from 3.3."
		exit;
	else
		echo -e "${orange}${bold}WARNING: ${normal}${NC}\tpython.${major:0:1}.${minor:0:1}: Python 3 is supported via 2to3 only"
	fi
fi
#check python version
#----------------------------------------------------------------------------------------------------------------------------------------------#
function check {
python - <<END
try:
    import numpy
except ImportError:
    print "Numpy doesn't seem to be installed. We will try to install it for you"
try:
    import scipy
except ImportError:
    print "Scipy doesn't seem to be installed. We will try to install it for you"
try:
    import theano
except ImportError:
    print "Theano doesn't seem to be installed. We will try to install it for you"
END
}
result=$(check)
if [[ $result != "" ]]; then
	echo -e "${red}${bold}Error:${normal}${NC}\t$result"
	echo -e "Choose your linux distribution"
	echo -e "\t1. Fedora."
	echo -e "\t2. Debian-Ubuntu."
	echo -e "\t3. Other."
	echo -e "\t4. Exit."
	read -p "Enter your choice [ 1 -4 ] " choice
	case $choice in
		1)
			sudo yum insall python-devel;
			sudo yum install f2py;
 			sudo yum install numpy scipy python-matplotlib ipython python-pandas sympy python-nose blas blas-devel python-pip
			sudo pip install Theano;sudo pip install --upgrade theano;
			echo -e "${green}${bold}\nDone${normal}${NC}\tNotes: http://deeplearning.net/software/theano/install.html"
			exit
			;;
		2)
			sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git python-pip;
			sudo pip install Theano;sudo pip install --upgrade theano;
			echo -e "${green}${bold}\nDone${normal}${NC}\tNotes: http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu"
			exit
			;;

		3)
			echo "Please follow instructions to install: http://www.scipy.org/scipylib/building/linux.html#specific-instructions"
			;;
		4)
			exit;;
		esac
		echo $(check)
else
	echo -e "${green}${bold}\nDone:${normal}${NC}\tEverything seems to be installed properly";
fi