echo Provide with the data path to store files:
read path
cd $path
wget https://www.openslr.org/resources/60/train-clean-360.tar.gz
tar -xvzf dev-clean.tar.gz