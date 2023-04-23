echo Provide with the data path to store files:
read path
cd $path
wget https://www.openslr.org/resources/60/dev-clean.tar.gz
tar -xvzf dev-clean.tar.gz