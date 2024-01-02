import random
import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r',encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for analogy
while True:
    input_term = input("\nEnter a word (EXIT to break): ").lower()
    list_a = {}
    if input_term == 'exit':
        break
    else:
        try:
            searched_word = vocab[input_term]
            for row, word in enumerate(W):
                distance = np.sqrt(np.sum((W[searched_word] - word) ** 2))
                if len(list_a) < 4:
                    list_a[row] = distance
                else:
                    for key, value in list_a.items():
                        if distance < value:
                            list_a[row] = list_a.pop(key)
                            list_a |= {row: distance}
                            break
            print(f"Four words closest to {input_term}")
            print("\n                            Word "
                  "         Distance")
            print("---------------------------------------------------")
            for key, value in dict(
                    sorted(list_a.items(), key=lambda item: item[1])).items():
                print("%32s\t\t  %f" % (ivocab[key], value))
        except KeyError:
            print(f"Word {input_term} could not be found")
        except IndexError:
            print("Wrong amoun of words")



