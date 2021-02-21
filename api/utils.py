import numpy as np
import pickle
import re
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS


def load_encoder(): 
    encoder = LabelEncoder()
    encoder.classes_ = np.load('../classes.npy')
    return encoder

def load_tokenizer():
    with open('../tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer

def clean_review(review, stopwords):
    html_tag = re.compile('<.*?>')
    cleaned_review = re.sub(html_tag, "", review).split()
    cleaned_review = [i for i in cleaned_review if i not in stopwords]
    return " ".join(cleaned_review)

def load_model():
    model = keras.models.load_model('../movieReview.model')
    model.load_weights('../MovieReview.h5')
    return model

def mapResult(ratio):
    res = np.rint(ratio)
    return "Postive" if res[0] else "Negatif"

def predictReview(review):
    max_len = 500
    rv = clean_review(review, stop_words)
    print('review text: {}\n'.format(review))
    tokenizer = load_tokenizer()
    tt = tokenizer.texts_to_sequences([rv])
    tt = pad_sequences(tt, maxlen=max_len)
    model = load_model()
    res = model.predict(tt)
    print("Predicted ---> {} \n\n".format(res))
    return mapResult(res.reshape(1))


if __name__ == '__main__':
    import json
    test = "The plot - in the future when nearly all men have been killed by a Y-chromosome-targeting virus, a (hot) female genetic engineer 'creates' a man in a chem lab - is intriguing. Despite the somewhat promising premise, the movie falls flat in nearly every regard. The dialogue is laughable. The characters are paper thin. The exploration of a single-gender world is shallow. The worst part of the entire movie is the Asian detective who delivers lines so cheesy and contrived that you'll want to vomit.<br /><br />I can't imagine how on earth this trash got produced. Most of the movie is male bashing. ""All men are violent."" ""All men rape women."" ""Men are only animals."" All of the women - even the 'closet hetero cases' - seem to display anger toward-, fear of-, and hatred for men. If you want to see a sci-fi film something along the lines of this movie's premise, you'd do best to look elsewhere"
    ovj = {"review" : test}
    f = open("demo.json", "a")
    f.write(json.dumps(ovj))
    f.close()
    print(predictReview(test))