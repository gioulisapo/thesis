textpath="~/.luarocks/share/lua/5.1/dp/data/mnist.lua";
i=0;
while read line
do
    name=$line
    ((i++))
    if [[ $name == *"USE_CUDNN"* ]]; then
    	echo "Enable cuDNN acceleration switch?"
	    read -p '[Y/n]: ' want_to_continue </dev/tty
	    case "${want_to_continue}" in
	    Y|y)
			 response=$i'c\USE_CUDNN\:=\1'
	         sed -i -r -e $response $textpath
	         continue;
	        ;;
	    *)
			response=$i'c\#USE_CUDNN\:=\1'
	        sed -i -r -e $response $textpath
	        continue;
	        ;;
	    esac
    elif [[ $name == *"CPU_ONLY"* ]]; then
    	echo "build without GPU support?"
	    read -p '[Y/n]: ' want_to_continue </dev/tty
	    case "${want_to_continue}" in
	    Y|y)
			 response=$i'c\CPU_ONLY\:=\1'
	         sed -i -r -e $response $textpath
	         continue;
	        ;;
	    *)
			 response=$i'c\#CPU_ONLY\:=\1'
	         sed -i -r -e $response $textpath
	        continue;
	        ;;
	    esac
    fi
done < $textpath

.luarocks/share/lua/5.1/dp/data/mnist.lua