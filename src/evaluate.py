import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse
from dotenv import load_dotenv

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD')

params = yaml.safe_load(open('params.yaml'))['train']


def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns="Outcome")
    y = data["Outcome"]
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    
    # load model from disk 
    model = pickle.load(open(model_path, 'rb'))
    
    predictions = model.predict(X)
    accuracy = accuracy_score(predictions,y)
    
    # log metrics to mlflow 
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")
    
if __name__ == "__main__":
    evaluate(params["data"], params["model"])
    
    