# MovieReview

Training in 2 epochs : 
![hg](imgs/ModelHistory.png)


#### Word Embedding 

-> Technique of natural language processing
-> Maps words or phrases to real numbers
-> Similar words that have the same meaning, have a similar representation

```
https://nlp.stanford.edu/projects/glove/
```



Convert a text into a sequence of nbs to put it into the neural network model

```python
# https://www.kaggle.com/hamishdickson/using-keras-oov-tokens
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_features, oov_token=oov)
tokenizer.fit_on_texts(reviews["cleaned_review"])
tokenized = tokenizer.texts_to_sequences(reviews["cleaned_review"])
```

For labels : 
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(df['review])
label_encoder = encoder.transform(df['review])
```


Split and make sure all the data are of the same X length: 
```python
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

XX = sequence.pad_sequences(tokenized, maxlen=X)
Xtrain, Xtest, ytrain, ytest = train_test_split(XX, label_encoder)
```

## API 

### Install requirements 
```
pip3 install -r requirements.txt
```

### Run API
```
cd api
python3 api.py
```

### Request :
Exemple using postman: 
```
link : http://127.0.0.1:5000/predict
JSON body: { "review": "The plot - in the future when nearly all men have been killed by a Y-chromosome-targeting virus, a (hot) female genetic engineer 'creates' a man in a chem lab - is intriguing. Despite the somewhat promising premise, the movie falls flat in nearly every regard. The dialogue is laughable. The characters are paper thin. The exploration of a single-gender world is shallow. The worst part of the entire movie is the Asian detective who delivers lines so cheesy and contrived that you'll want to vomit.<br /><br />I can't imagine how on earth this trash got produced. Most of the movie is male bashing. All men are violent. All men rape women. Men are only animals. All of the women - even the 'closet hetero cases' - seem to display anger toward-, fear of-, and hatred for men. If you want to see a sci-fi film something along the lines of this movie's premise, you'd do best to look elsewhere" }

Response :  
{
    "result": "Negatif"
}

```

TODO: 
```
Run it in a Docker container
```