import flask
from flask import jsonify, request
from utils import predictReview

app = flask.Flask(__name__)
app.config["DEBUG"] = True




@app.route('/', methods=['GET'])
def home():
    return "<h1>mMvie review sentiment classification</h1><p>This is a test, call /predict.</p>"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    body = request.json
    if 'review' in body:
        print('\n\n\n{}\n\n\n'.format(body['review']))
        res = predictReview(body['review'])
        return jsonify({'result': res}) 
    return jsonify({'result': "ERROR"}) 

app.run()
