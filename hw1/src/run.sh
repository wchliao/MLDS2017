
if (( $# != 2 )); then
    echo "Illegal number of parameters"
    exit
fi

#wget https://dl.dropboxusercontent.com/u/16850414/model.tar.gz
#tar xzvf model.tar.gz
python ptb_word_lm_final.py --q_path=$1 --save_path=../../../model_old --p_path=$2
