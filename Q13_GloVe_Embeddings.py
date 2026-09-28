import numpy as np

def load_glove(file_path):
    embeddings = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector

    return embeddings


glove = load_glove("glove.6B.50d.txt")

words = ["king", "queen", "computer", "apple"]

for word in words:
    if word in glove:
        print("\nWord:", word)
        print("Vector:", glove[word])
    else:
        print("\nWord:", word)
        print("Word not found")
