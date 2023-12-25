import requests
import json
import re
import string
import numpy as np
import joblib
from models.transformerSS import CustomizedModel
# from models.CNN_model import CNNModel

class TextClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TextClassifier, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, models_directory='models'):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.cleaner = TextCleaner()
        self.models = {}
        self.load_models(models_directory)

    def load_models(self, models_directory):
        self.models['Logistic Regression'] = joblib.load(models_directory + '/ML_models/logreg_model.pkl')
        self.models['Naive Bayes'] = joblib.load(models_directory + '/ML_models/nb_model.pkl')
        self.models['SVM'] = joblib.load(models_directory + '/ML_models/svm_model.pkl')
        # self.models['CNN'] = CNNModel()
        self.models['Fine-Tuned DistilBert'] = CustomizedModel(models_directory="models/transformer_model")

    def print():
        print('hello classifier')

    def clean_text_input(self, text):
        cleaned_text = self.cleaner.clean_text(text)
        return cleaned_text

    def predict(self, text, model_name):
        print("predict called.....")
        cleaned_text = self.clean_text_input(text)
        model = self.models[model_name]
        if model_name == 'Fine-Tuned DistilBert':
            print("Gooooo ")
            return self.models[model_name].predict(cleaned_text)
        elif (model_name == 'CNN'):
            return self.models[model_name].predict(cleaned_text)
        else:
            latent_representation = self.models['Fine-Tuned DistilBert'].extract_dense_128_vectors(cleaned_text)
            prediction = model.predict(latent_representation.reshape(1, -1))
            probabilities = {}
            print(prediction[0])
            if prediction[0] == 0:
                probabilities = {"fake": 1, "real": 0}
            else:
                probabilities = {"fake": 0, "real": 1}
            prediction = int(prediction)

        response = {
            "text": text,
            "selected_model": model_name,
            "result": prediction,
            "probabilities": probabilities
        }

        return response

                
class TextCleaner:
    def __init__(self):
        self.regex_pattern = re.compile('[%s]' % re.escape(string.punctuation))

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]',' ', text)  # Remove text in square brackets
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Remove links
        text = re.sub(r'<.*?>+', ' ', text)  # Remove HTML tags
        text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)  # Remove punctuation
        text = re.sub(r'\n', ' ', text)  # Remove newline characters
        text = re.sub(r'\w*\d\w*', ' ', text)  # Remove words containing numbers
        text = re.sub(r'[.]', ' ', text)  # Remove period
        return text