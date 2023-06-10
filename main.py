from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k
import numpy as np
import json
import uuid
import requests
import os
from flask_mysqldb import MySQL
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

tf.keras.backend.clear_session()
app = Flask(__name__)
CORS(app)

# MySQL configurations
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'your_username')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'your_password')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'your_database_name')

mysql = MySQL(app)

def preprocess_image(image):
    desired_size = (64, 64)
    image = np.resize(image, desired_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def create_model():
    inputs = Input((64, 64, 1))
    x = Conv2D(64, (11, 11), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (7, 7), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    pooledOutput = Dense(512)(pooledOutput)
    outputs = Dense(128)(pooledOutput)

    model = Model(inputs, outputs)
    return model

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

feature_extractor = create_model()
imgA = Input(shape=(64, 64, 1))
imgB = Input(shape=(64, 64, 1))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)
distance = Lambda(euclidean_distance)([featA, featB])
outputs = Dense(1, activation="sigmoid")(distance)

@app.route('/users', methods=['GET'])
def get_users():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    data = cur.fetchall()
    cur.close()
    users = []
    for user in data:
        user_dict = {
            'id': user[0],
            'name': user[1],
            'email': user[2]
        }
        users.append(user_dict)
    return jsonify(users)

@app.route('/users', methods=['POST'])
def add_user():
    name = request.json['name']
    email = request.json['email']

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO users (name, email) VALUES (%s, %s)", (name, email))
    mysql.connection.commit()
    cur.close()

    return jsonify({'message': 'User added successfully'})

@app.route('/training', methods=['POST'])
def train():
    # Your training code here
    return jsonify({'message': 'Training endpoint'})

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction code here
    return jsonify({'message': 'Prediction endpoint'})

if __name__ == '__main__':
    app.run(debug=True)
