from gensim.models.word2vec import Word2Vec, KeyedVectors
from random import randint,seed
seed(42)
from multiprocessing import cpu_count
model = KeyedVectors.load("semantle")
#model = Word2Vec.load('semantle')
#print(model.wv.most_similar('renaissance'))
#print(model.wv.most_similar('resurrection'))
def rng_word():
    while True:
        i = randint(0, len(model.index_to_key))
        w = model.index_to_key[i]
        if w.isalpha() and w.islower():
            return w,i

# main loop
while True:
    w,i = rng_word()
    #print(f"Suggested random word: {w}")
    w = input("Word: ")
    s = float(input("Score: "))
    # iterate through and print the ones closest in score?
    for word in model.key_to_index.keys():
        if word.isalpha() and word.islower():
            similarity = model.similarity(w,word)
            dist = 2*(1-similarity)
            score = similarity*100.0
            if abs(s-score) < .001:
                print(word, score, dist)
