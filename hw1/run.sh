
if (( $# != 2 )); then
    echo "Illegal number of parameters"
    exit
fi

wget https://dl.dropboxusercontent.com/u/16850414/model.tar.gz
tar xzvf model.tar.gz

Q=$1
P=$2
M=$(pwd)/model_old

cd src
python ptb_word_lm_final.py --q_path=$Q --save_path=$M --p_path=$P
