import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import json
# # Provide the path of the saved model
# model_path = "path/to/saved/model.h5"


class CNNModel:
    def __init__(self, model_path="/CNN/model.h5"):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = models.load_model(self.model_path)
        return model

    def preprocess_input(self, input_vector):
        input_matrix = input_vector.reshape((8, 16))
        input_reshaped = np.reshape(input_matrix, (1, 8, 16, 1))
        return input_reshaped

    def predict(self, input_vector):
        input_reshaped = self.preprocess_input(input_vector)
        prediction = self.model.predict(np.expand_dims(input_reshaped, axis=0))
        print(prediction)
        probabilities = prediction[0]
        prediction_class = np.argmax(prediction, axis=1)[0]
     
        # Prepare the probabilities as a dictionary
        probabilities_dict = { "Fake": float(probabilities[0]), "True": float(probabilities[1]) }
        
        # Prepare the response as a JSON object
        response = {
            "result": result,
            "probabilities": probabilities_dict
        }
        
        return response