import random
import traceback

import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# W contains vectors for
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v

# Main loop for analogy
while True:
    input_term = input("\nEnter three words (EXIT to break): ").lower()
    list_a = {}
    if input_term == 'exit':
        break
    else:

        try:
            input_term = input_term.split(" ")

            x = vocab[input_term[0]]
            y = vocab[input_term[1]]
            z = vocab[input_term[2]]

            answer = W[z] + (W[y] - W[x])

            for row, word in enumerate(W):
                distance = np.sqrt(np.sum((answer - word) ** 2))
                if len(list_a) < 3:
                    list_a[row] = distance
                else:
                    for key, value in list_a.items():
                        if distance < value:
                            list_a[row] = list_a.pop(key)
                            list_a |= {row: distance}
                            break

            print(f"""Three best words to the analogy \n"{input_term[0]} """
                  f"""is to {input_term[1]} as {input_term[2]} is to X" """)
            print("\n                            Word ",
                  "        Distance")
            print(
                "------------------------------------------------------")

            for key, value in dict(
                    sorted(list_a.items(), key=lambda item: item[1])).items():
                print("%32s\t\t  %f" % (ivocab[key], value))
        except IndexError:
            print("Wrong number of words")
        except KeyError as e:
            print(f"Word {e} was not found")
