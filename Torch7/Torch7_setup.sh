#!/bin/bash

#-------------------install----------------------------------------------------------------------------------

#Method_1: works on MacOS X 10.8, Ubuntu 12.04 and Fedora 20 and earlier
#sudo curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | zsh

#Method_2: Tested in Ubuntu_14.04LTS
sudo curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash;
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
#In case the error occurs while linking Linking C executable luajit
#1) Delete ~/torch and re-clone the project in ~/torch
#2) Edit ~/torch/install.sh (49): cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release -DWITH_LUAJIT21=ON
#						------>c: cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release
# Run ~/torch/install.sh

#------------install-extras----------------------------------------------------------------------------------
luarocks install image  && luarocks install nnx      # lots of extra neural-net modules

#-------------------uninstall----------------------------------------------------------------------------------
#sudo rm -rf ~/torch