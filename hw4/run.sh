#!/bin/bash
# bash run.sh S2S input.txt prediction.txt

if [ $s1 -eq "S2S" ]
then
	echo "> RUNNING S2S"
	python3 src/download_models_s2s.py
	python3 src/main.py --mode test --model_name open_subtitles --vocab_size 80000 --size 128 --antilm 0.7 --n_bonus 0.1 --rev_model 0 --reinforce_learn 0 --input_name $s2 --output_name $s3
elif [ $s1 -eq "RL" ]
then
	echo "> RUNNING RL"
	python3 src/download_models_rl.py
	python3 src/main.py --mode test --model_name open_subtitles_best --vocab_size 80000 --size 128 --antilm 0.7 --n_bonus 0.1 --rev_model 1 --reinforce_learn 1 --input_name $s2 --output_name $s3
elif [ $s1 -eq "BEST" ]
then
	echo "> RUNNING BEST"
	python3 src/download_models_s2s.py
	python3 src/main.py --mode test --model_name open_subtitles_best --vocab_size 80000 --size 128 --antilm 0.7 --n_bonus 0.1 --rev_model 1 --reinforce_learn 1 --input_name $s2 --output_name $s3
else
	echo "> Please only use {S2S|RL|BEST} as first argument"
fi