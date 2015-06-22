!/bin/zsh


#
#Requires cmake, qt 4.x,
#Must run with root privilages
#----install----------------------------------------------------------------------------------
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | zsh;
path="../../frameworks/torch";
git clone https://github.com/torch/distro.git $path --recursive && cd $path; ./install.sh && rmdir temp && luarocks install image  && luarocks install nnx      # lots of extra neural-net modules