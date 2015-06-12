green='\033[0;32m'
orange='\033[0;33m'
red='\033[0;31m'
NC='\033[0m' # No Color
bold=`tput bold`
normal=`tput sgr0`

#Check if Java version is Greater than 1.7
if type -p java; then
    _java=java
elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]];  then   
    _java="$JAVA_HOME/bin/java"
else
    echo "No version of java was found in your system please install Java\n"
    return;
fi

if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [[ "$version" < "1.7" ]]; then
        echo version is less than 1.7 please update
        return;
    else         
        echo -e "Java.$version ${green}${bold}\tOK${normal}${NC}"
    fi
fi

#Check if maven is installed
MavenCheck=`mvn -version`
if [[ $MavenCheck == *"Warning"* ]]; then
    echo -e "${orange}$MavenCheck${NC}"
elif [[ $MavenCheck == *"Error"* ]]; then
    echo -e "${red}$MavenCheck${NC}"
    return;
elif [[ $MavenCheck == "" ]]; then
    echo -e "${red}Install Maven${NC} run: ${orange}Debian:  ${NC}sudo apt-get install maven\n\t\t${orange}   Red-Hat: ${NC}sudo yum install maven"
    exit;
else
    echo -e "Maven\t\t${green}${bold}OK${normal}${NC}"    
fi
#Check if Blas library is installed
blas=`ls /usr/lib64/libblas.so*`
if [[ $blas == "" ]]; then
    echo -e "${red}Install Blas${NC} run: ${orange}Debian:  ${NC}sudo apt-get install libblas*\n\t\t${orange}  Red-Hat: ${NC}sudo yum -y install blas"
    exit;
else
    echo -e "Blas\t\t${green}${bold}OK${normal}${NC}"
fi
#git clone https://github.com/SkymindIO/nd4j
#git clone https://github.com/SkymindIO/deeplearning4j
#git clone https://github.com/SkymindIO/nd4j-examples
#git clone https://github.com/SkymindIO/dl4j-examples