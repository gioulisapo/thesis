#!/bin/bash
git clone https://github.com/BVLC/caffe.git;
mv caffe/ ../../frameworks/
for req in $(cat ../../frameworks/caffe/python/requirements.txt); do sudo pip install $req; done

#libgoogle-glog-dev required && libatlas-base-dev && protobuf-compiler libprotobuf-dev libprotoc-dev libtiff5-dev:i386 libtiff5-dev 
#build-essential libxml2-dev libgeos++-dev libpq-dev libbz2-dev proj libtool automake 
#libprotobuf-c0-dev protobuf-c-compiler && libopencv-dev && libhdf5-serial-dev && hdf5-tools
#Manually install ffmpeg and OpenVC down vote For ffmpeg:  ./configure --enable-gpl --enable-libfaac --enable-libmp3lame --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libtheora --enable-libvorbis --enable-libxvid --enable-nonfree --enable-postproc --enable-version3 --enable-x11grab
#make  #sudo make install #For OpenCV,#cmake -D CMAKE_BUILD_TYPE=RELEASE ..#make #sudo make install
#If compiler error /usr/bin/ld: cannot find -lhdf5_hl --> sudo ln -s /usr/lib/libhdf5.so.6 /usr/lib/libhdf5.so
#if compiler error /usr/bin/ld: cannot find -lhdf5------> sudo ln -s /usr/lib/libhdf5.so.6 /usr/lib/libhdf5.so
#sudo apt-get install libv4l-dev
#cd /usr/include/linux
#sudo ln -s ../libv4l1-videodev.h videodev.h

textpath="../../frameworks/caffe/Makefile.config.example";
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
cp $textpath ../../frameworks/caffe/Makefile.config;
cd ../../frameworks/caffe/ && make all && make test && make runtest;
