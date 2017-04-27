wget https://www.dropbox.com/s/qc4nbo86fqryn6p/model.tar.gz
tar zxvf model.tar.gz
python ./src/attention.py $1 $2 --test

