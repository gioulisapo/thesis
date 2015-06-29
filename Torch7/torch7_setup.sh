#!/bin/bash

#-------------------install----------------------------------------------------------------------------------

#Method_1: works on MacOS X 10.8, Ubuntu 12.04 and Fedora 20 and earlier
#sudo curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | zsh

#Method_2: Tested in Ubuntu_14.04LTS
sudo curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash;
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
#In case an error occurs while linking Linking C executable luajit
	#1) Delete ~/torch and re-clone the project in ~/torch
	#2) Edit ~/torch/install.sh (49): cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release -DWITH_LUAJIT21=ON
	#						------>c: cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release
	#3) Run ~/torch/install.sh
# 
#In case while trying to run torch (by executing th) the following error appears $HOME/torch/install/bin/luajit:\
#	symbol lookup error: $HOME/.luarocks/lib/lua/5.1/libtorch.so: undefined symbol: luaT_setfuncs
#	#1) delete the ~/.luarocks folder
	#2) make sure lua5.1 is installed in your system
	#3) download and utar the latest version of luarocks (from http://keplerproject.github.io/luarocks/releases/) into ~/.luarocks
	#4) cd ~/.luarocks && ./configure && make build && sudo make install


#------------install-extras----------------------------------------------------------------------------------
luarocks install dp --local   # lots of extra neural-net modules

#-------------------uninstall----------------------------------------------------------------------------------
#sudo rm -rf ~/torch