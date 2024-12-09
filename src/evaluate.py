import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/bimbo-22/machine_learning_pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "bimbo-22"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "ecab5a821a111559ec1e2a91a6a19249fed246e7"

params = yaml.safe_load(open('params.yaml'))['train']


def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns="Outcome")
    y = data["Outcome"]
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    
    # load model from disk 
    model = pickle.load(open(model_path, 'rb'))
    
    predictions = model.predict(X)
    accuracy = accuracy_score(predictions,y)
    
    # log metrics to mlflow 
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")
    
if __name__ == "__main__":
    evaluate(params["data"], params["model"])
    
    