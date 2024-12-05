import nltk
import pandas as pd
from nltk.corpus import wordnet

nltk.download('wordnet')

def generate_one_word_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if ' ' not in lemma.name() and '_' not in lemma.name() and '-' not in lemma.name():
                synonyms.add(lemma.name())
    return list(synonyms)

genres = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Family', 'Fantasy',
    'Fiction', 'Foreign', 'History', 'Horror', 'Movie', 
    'Music', 'Mystery', 'Romance', 'Science', 'TV', 'Thriller', 'War', 'Western'
]

data = []

for genre in genres:
    synonyms = generate_one_word_synonyms(genre.lower())  
    for synonym in synonyms:
        data.append([genre.lower(), synonym])  

df = pd.DataFrame(data, columns=['labels', 'features'])

df.to_csv('genre_one_word_synonyms.csv', index=False)

print("One-word synonyms have been saved to 'genre_one_word_synonyms.csv'.")
