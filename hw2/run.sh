wget https://www.dropbox.com/s/y2b6pz7ksqnijcs/model.tar.gz
tar zxvf model.tar.gz
python ./src/s2vt.py $1 $2 --test

