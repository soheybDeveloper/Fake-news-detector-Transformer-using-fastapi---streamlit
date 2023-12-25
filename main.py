from fastapi import FastAPI

from pydantic import BaseModel
import pickle
import requests
import responses
import json
from classifiers import TextClassifier


app = FastAPI()


@app.post("/predict")
async def predict_text(input_data: dict):
    try:
        text = input_data["data"]
        model_selected = input_data["selected_model"]
        clf = TextClassifier()
        prediction = clf.predict(text, model_selected)
        print(prediction)
  
        return prediction
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": "Something went wrong"}

@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}
 















# @app.post("/process")
# def process_data(input_data: MyInputModel):
#     # Generate latent representation using the transformer model
#     encoded_input = tokenizer(input_data.data, padding=True, truncation=True, return_tensors="pt")
#     latent_representations = transformer_model(**encoded_input).last_hidden_state

#     # Perform predictions using the ML model
#     result = ml_model.predict(latent_representations)

#     # Return the processed result
#     return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
