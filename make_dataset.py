from sklearn.utils import shuffle
from keras.preprocessing import sequence
import nltk
import itertools
import pickle
import numpy as np

VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 20

confusion_pairs = [['edition', 'addition'],
['shall', 'shell'],
['see', 'sea'],
['role', 'roll'],
['isle', 'aisle'],
['your' , "you're"],
['which', 'witch'],
['wrong', 'wring'],
['wrongs', 'wrings'],
['things', 'thinks'],
["dont", "don't"]]

def find_list(word) :
	for lis in confusion_pairs :
		if word in lis :
			return lis

def find_pairs(found_words,depth) :
	if depth == len(found_words) -1 :
		temp_list =  find_list(found_words[depth])
		return [[word] for word in temp_list]

	else :
		next_list = find_pairs(found_words,depth+1)
		now_list = find_list(found_words[depth])
		new_list = []
		for i in now_list :
			for j in next_list :
				temp_list = []
				temp_list.append(i)
				for k in j :
					temp_list.append(k)
				new_list.append(temp_list)
		return new_list

confusion_words = [word for lis in confusion_pairs for word in lis]

count = 0 

eng_sentences = []

print("[INFO] -> Reading Training File")

with open("eng_sent.txt") as f:
	for line in f.readlines():
		line = line.replace(".","").replace("!","").replace("?","").replace(",","").lower()
		for word in line.split() :
			if word in confusion_words :
				eng_sentences.append(line)
				break

print("[INFO] -> Done ")

training_data = []
training_targets = []
for sent in eng_sentences :
	position = []
	found_words = []
	for i,word in enumerate(sent.split()) :
		if word in confusion_words :
			position.append(i)
			found_words.append(word)

	pairs = find_pairs(found_words,0)

	for pair in pairs :
		data = []
		target = []
		for i,word in enumerate(sent.split()):
			if i not in position :
				data.append(word)
				target.append("0")
			else:
				data.append(pair[position.index(i)])
				if pair[position.index(i)] == word :
					target.append("0")
				else :
					target.append("1")
		training_data.append(" ".join(data))
		training_targets.append(" ".join(target))

training_data = np.array(training_data)
training_targets = np.array(training_targets)

training_data,training_targets = shuffle(training_data,training_targets)

input_list = []
output_list = []

with open("training_data.txt",'w') as f:
	for line in training_data :
		input_list.append(line.split())
		f.write(line + '\n')

with open("training_targets.txt",'w') as f:
	for line in training_targets :
		mod_line = "2 "+line+" 3"
		output_list.append(mod_line.split())
		f.write(line + '\n')
		
temp_input = [word for k in input_list for word in k ]
input_word_freq = nltk.FreqDist(itertools.chain(temp_input))
input_vocab = input_word_freq.most_common(VOCAB_SIZE-1)

temp_output = [word for k in output_list for word in k ]
output_word_freq = nltk.FreqDist(itertools.chain(temp_output))
output_vocab = output_word_freq.most_common(VOCAB_SIZE-1)

input_index_to_word = [x[0] for x in input_vocab]
input_index_to_word.append("UNK")
word_to_index = dict([(w,i) for i,w in enumerate(input_index_to_word)])

for i, sent in enumerate(input_list):
    input_list[i] = [w if w in word_to_index.keys() else "UNK" for w in sent]

X = np.asarray([[word_to_index[w] for w in sent] for sent in input_list])
Y = np.asarray([[w for w in sent] for sent in output_list])

padded_input = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
padded_output = sequence.pad_sequences(Y, maxlen=MAX_SEQUENCE_LENGTH, padding='post',value = 3)


f = open("./pickle_files/word_to_index.pkl" , 'wb')
pickle.dump(word_to_index,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()

f = open("./pickle_files/padded_input.pkl" , 'wb')
pickle.dump(padded_input,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()

f = open("./pickle_files/padded_output.pkl" , 'wb')
pickle.dump(padded_output,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()

f = open("./pickle_files/selected_confusion_pairs.pkl" , 'wb')
pickle.dump(confusion_pairs,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()

print("[INFO] -> Preprocessing Done")
