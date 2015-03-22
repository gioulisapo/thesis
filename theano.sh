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
function current_datetime {
python - <<END
try:
    import numpy
except ImportError:
    print "error"
try:
    import scipy
except ImportError:
    print "error"
END
}
result=$(current_datetime)
if [[ $result == "error" ]]; then
	echo -e "${red}${bold}Error:${normal}${NC}\tnumpy/scipy is not installed"
	echo "Choose your linux distribution"
	echo "1. Fedora."
	echo "2. Debian-Ubuntu."
	echo "3. Gentoo."
	echo "4. Other."
	echo "5. Exit."
	read -p "Enter your choice [ 1 -5 ] " choice
	case $choice in
		1)
			sudo yum install numpy scipy python-matplotlib ipython python-pandas sympy python-nose
			;;
		2)
			sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
			;;
		3)
			sudo emerge -aN '>=dev-python/numpy-1.6' '>=sci-libs/scipy-0.10' '>=dev-python/matplotlib-1.1' '>=dev-python/ipython-0.13' '>=dev-python/pandas-0.8' '>=dev-python/sympy-0.7' '>=dev-python/nose-1.1'
			;;
	
		4)
			echo "Please follow instructions to install: http://www.scipy.org/scipylib/building/linux.html#specific-instructions"
			;;
		5)
			exit;;
		esac
else
	echo -e "${green}${bold}OK${normal}${NC}\tNumpy installed"
fi