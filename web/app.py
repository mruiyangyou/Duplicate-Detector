import json
from flask import render_template, request, Flask
import pandas as pd
import numpy as np
import pip
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import Dataset
import os
# print(os.getcwd())

trained_model_path = 'bestmodel'
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(trained_model_path, num_labels = num_labels)
model_checkpoint = "distilbert-base-uncased"
pipeline = TextClassificationPipeline(model = model, 
                                    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_legnth = 512))
tokenize_kwargs = {'padding': 'max_length', 'truncation': True,
                'max_length': 512}


def predict(text):
    pred = pipeline(text, **tokenize_kwargs)[0]
    pred['label'] = 'duplicate' if int(pred['label'][6]) == 1 else 'Not duplicate'
    return pred

app = Flask(__name__)

@app.route('/')
def toIndexPage():
    return render_template('index.html')

@app.route('/cal')
def classify():
    data = request.args
    text = data['input1']
    res = predict(text)
    return json.dumps(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

# if __name__ == '__main__':
#     app.run()