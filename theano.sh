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
elif [[ ${major:0:1} -ge 2 ]]; then
	#statements
	if [ ${minor:0:1} -ge 6 ]; then
		echo -e "${green}${bold}OK${normal}${NC}\tpython.${major:0:1}.${minor:0:1}"
	else
		echo -e "${red}Theano requires python.2.6 or more;"
		exit;
	fi
else
	if [ ${minor:0:1} -le 2 ]; then
		echo -e "${red} Python 3 is supported via 2to3 only, starting from 3.3."
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
    print "numpy is not installed"
END
}
result=$(current_datetime)
if [[ $result == "numpy is not installed" ]]; then
	echo "numpy is not installed please follow instructions to install: http://www.scipy.org/scipylib/building/linux.html"
	exit;
else
	echo -e "${green}${bold}OK${normal}${NC}\tNumpy installed"
fi