import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
import ast
import datetime

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)) 
    image = preprocess_input(image)  
    return image

def extract_features(image):
    features = model.predict(np.expand_dims(image, axis=0))
    return features.flatten()

def add_image_feature(image_path):
    image_name = image_path.split("\\")[-1].split('.')[0]  
    image = preprocess_image(image_path)
    features = extract_features(image)
    feature_str = ",".join(map(str, features))
    file_exists = os.path.isfile('feature_data.csv')
    new_data = pd.DataFrame([[image_name, feature_str]], columns=["filename", "features"])
    new_data.to_csv('feature_data.csv', mode='a', index=False, header=not file_exists)

    return {'outcom': f"Added {image_name} to {'feature_data.csv'}"}


def compute_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]


def match_fingerprint(new_fingerprint_path, threshold=0.9):

    new_features = extract_features(preprocess_image(new_fingerprint_path))
    
    best_match = None
    highest_similarity = -1  
    dataf=pd.read_csv("feature_data.csv")
    
    for index, row in dataf.iterrows():
        similarity = compute_similarity(new_features, ast.literal_eval(row['extract_features']))
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = row['image_name']
    
    if highest_similarity >= threshold:
        return {
            'id':best_match,
            'Similarity':highest_similarity,
            'time':datetime.datetime.now(),
            'note':None
        }
    else:
        return {
            'id':None,
            'Similarity':None,
            'time':datetime.datetime.now(),
            'note':'No person found. Fingerprint is not recognized . try again'
        }
