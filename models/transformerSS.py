import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertModel, DistilBertConfig
# import os
import torch.nn as nn
import torch.nn.functional as F

class CustomizedModel:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CustomizedModel, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, models_directory):
        if self.initialized:
            return
        self.initialized = True
        self.models_directory = models_directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.models_directory)
        self.model = self.load_model_with_additional_layers(self.models_directory)


    def load_model_with_additional_layers(self, models_directory):
        print()
        print()
        print("loading the transformer")
        print()
        base_model = AutoModelForSequenceClassification.from_pretrained(models_directory)

        # additional_layers_path = f"{models_directory}/additional_layers.pth"
        # additional_layers = torch.load(additional_layers_path, map_location=torch.device('cpu'))

        # num_labels = additional_layers['num_labels']
        # dense_128_state_dict = additional_layers['dense_128_state_dict']
        # dense_2_state_dict = additional_layers['dense_2_state_dict']
        # dropout_state_dict = additional_layers['dropout_state_dict']
        # classifier_state_dict = additional_layers['classifier_state_dict']

        # base_model.resize_token_embeddings(len(self.tokenizer))

        # base_model.dense_768 = nn.Linear(768, 768)
        # base_model.dense_2 = nn.Linear(768, 2)
        # base_model.dropout = nn.Dropout(0.3)
        # base_model.classifier = nn.Linear(2, num_labels)

        # base_model.dense_128.load_state_dict(dense_128_state_dict)
        # base_model.dense_2.load_state_dict(dense_2_state_dict)
        # base_model.dropout.load_state_dict(dropout_state_dict)
        # base_model.classifier.load_state_dict(classifier_state_dict)

        return base_model


    def forward(self, input_ids, attention_mask):
       
        distilbert_output = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output.last_hidden_state

        dense768_output = self.model.pre_classifier(hidden_state[:, 0, :])
        result = self.model.classifier(dense768_output)
        logits = self.model.dropout(result)

        return logits

    def predict(self, text):
        encoded_input = self.tokenize(text)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            print(logits)
            probabilities = torch.softmax(logits, dim=1)
        print("logits ",logits)
        print()
        print("probabilities ",probabilities)
        
        predicted_class_probabilities = probabilities[0].tolist()
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probability_dict = {"fake": predicted_class_probabilities[0], "real": predicted_class_probabilities[1]}

        probability_dict = {"fake": predicted_class_probabilities[0], "real": predicted_class_probabilities[1]}

        prediction = {"text": text, "selected_model": "DistilBERT","result": predicted_class,"probabilities": probability_dict
            }

        return prediction
      
    def tokenize(self, text):
        return self.tokenizer.encode_plus(
            text,
            padding='longest',
            truncation=True,
            max_length=512,
            # max_length=1024,
            return_tensors='pt'
      )
    def extract_dense_128_vectors(self, input_text):
        encoded_input = self.tokenize(input_text)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        with torch.no_grad():
            distilbert_output = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = distilbert_output.last_hidden_state

            dense_output = self.model.pre_classifier(hidden_state[:, 0, :])
            # print(dense_128_output.shape)
            dense_output = dense_output.cpu()
            # print(dense_output.shape)
        return dense_output.squeeze().detach().numpy().squeeze()











# import torch
# from transformers import AutoModel, AutoTokenizer, DistilBertModel, DistilBertConfig
# import os
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomizedModel:
#     _instance = None
#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(CustomizedModel, cls).__new__(cls)
#             cls._instance.initialized = False
#         return cls._instance

#     def __init__(self, models_directory):
#         if self.initialized:
#             return
#         self.initialized = True
#         self.models_directory = models_directory
#         self.tokenizer = AutoTokenizer.from_pretrained(self.models_directory)
#         self.model = self.load_model_with_additional_layers(self.models_directory)


#     def load_model_with_additional_layers(self, models_directory):
#         print()
#         print()
#         print("loading the transformer")
#         print()
#         base_model = AutoModel.from_pretrained(models_directory)

#         additional_layers_path = f"{models_directory}/additional_layers.pth"
#         additional_layers = torch.load(additional_layers_path, map_location=torch.device('cpu'))

#         num_labels = additional_layers['num_labels']
#         dense_128_state_dict = additional_layers['dense_128_state_dict']
#         dense_2_state_dict = additional_layers['dense_2_state_dict']
#         dropout_state_dict = additional_layers['dropout_state_dict']
#         classifier_state_dict = additional_layers['classifier_state_dict']

#         # base_model.resize_token_embeddings(len(self.tokenizer))

#         # base_model.dense_768 = nn.Linear(768, 768)
#         # base_model.dense_2 = nn.Linear(768, 2)
#         # base_model.dropout = nn.Dropout(0.3)
#         # base_model.classifier = nn.Linear(2, num_labels)

#         # base_model.dense_128.load_state_dict(dense_128_state_dict)
#         # base_model.dense_2.load_state_dict(dense_2_state_dict)
#         # base_model.dropout.load_state_dict(dropout_state_dict)
#         # base_model.classifier.load_state_dict(classifier_state_dict)

#         return base_model


#     def forward(self, input_ids, attention_mask):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[0][:, 0, :]  
#         dense_128_output = F.relu(self.model.dense_128(pooled_output))
#         dense_128_output = self.model.dropout(dense_128_output)
#         dense_2_output = F.relu(self.model.dense_2(dense_128_output))
#         dense_2_output = self.model.dropout(dense_2_output)
#         logits = self.model.classifier(dense_2_output)
#         return logits

#     def predict(self, text):
#         encoded_input = self.tokenize(text)
#         input_ids = encoded_input['input_ids']
#         attention_mask = encoded_input['attention_mask']

#         with torch.no_grad():
#             logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
#             print(logits)
#             probabilities = torch.softmax(logits, dim=1)
#         print("logits ",logits)
#         print()
#         print("probabilities ",probabilities)
        
#         predicted_class_probabilities = probabilities[0].tolist()
#         predicted_class = torch.argmax(probabilities, dim=1).item()
#         probability_dict = {"fake": predicted_class_probabilities[0], "real": predicted_class_probabilities[1]}

#         probability_dict = {"fake": predicted_class_probabilities[0], "real": predicted_class_probabilities[1]}

#         prediction = {"text": text, "selected_model": "DistilBERT","result": predicted_class,"probabilities": probability_dict
#             }



#         return prediction
      
#     def tokenize(self, text):
#         return self.tokenizer.encode_plus(
#             text,
#             padding='longest',
#             truncation=True,
#             max_length=512,
#             return_tensors='pt'
#       )
#     def extract_dense_128_vectors(self, input_text):
#         encoded_input = self.tokenize(input_text)
#         input_ids = encoded_input['input_ids']
#         attention_mask = encoded_input['attention_mask']

#         with torch.no_grad():
#             dense_128_output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
#             # print(dense_128_output.shape)
#             dense_output = self.model.dense_128(dense_128_output)
#             dense_output = dense_output.cpu()
#             # print(dense_output.shape)
#         return dense_output.squeeze().detach().numpy().squeeze()
