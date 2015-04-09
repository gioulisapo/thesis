#!/bin/bash

#----install----------------------------------------------------------------------------------
#curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash;
#path="../../frameworks/torch"
#git clone https://github.com/torch/distro.git $path --recursive;
#cd $path; ./install.sh
#rmdir temp;
#luarocks install image    # an image library for Torch7
luarocks install nnx      # lots of extra neural-net modules