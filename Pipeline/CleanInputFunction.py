import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
import numpy as np
import re

custom_words_to_filter = []

lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = nltk.corpus.stopwords.words('english')

def clean_input(data):
    data['title_stance'] = data.apply(lambda row : clean_text(row['title'], filter_stop_words=False), axis=1)
    data['title_topic'] = data.apply(lambda row : clean_text(row['title'], filter_stop_words=True), axis=1)

    return data

def clean_text(text, filter_stop_words=True):
    if filter_stop_words:
        words_to_filter = np.concatenate((stop_words, custom_words_to_filter))
    else:
        words_to_filter = custom_words_to_filter

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) # replace punctuation/special chars with a space (space instead of blank to prevent update 2-Hong Kong from becoming 2Hong)
    text = " ".join(text.split()) # remove extra whitespace
    text = lemmatizer.lemmatize(text)
    text = tokenizer.tokenize(text)
    text = [word for word in text if word not in words_to_filter]
    text = ' '.join(text)

    return text