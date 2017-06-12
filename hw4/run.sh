p1=data/movie_conversations.txt
p2=data/movie_lines.txt
p3=data/open_subtitles.txt
p4=data/chat.txt

processed_data=data/conversations.npy
dictionary=data/dictionary.txt
input=question.txt
output=result.txt

#python DataPreprocessor.py ${p1} ${p2} ${p3} ${processed_data} ${dictionary}
python seq2seq.py --train -t ${processed_data} -d ${dictionary}
#python seq2seq.py --test -q ${input} -o ${output} -d ${dictionary}
