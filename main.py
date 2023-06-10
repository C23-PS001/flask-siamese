from urllib.parse import unquote
from io import BytesIO
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from google.cloud import storage

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as k
import cv2
import numpy as np
import json
import uuid
import requests
import os

tf.keras.backend.clear_session()

app = Flask(__name__)
CORS(app)
def preprocess_image(image):
    # Resize the image to match the input size of your model
    desired_size = (64, 64)  # Replace with your model's input size
    image = cv2.resize(image, desired_size)

    # Normalize the pixel values
    image = image.astype('float32') / 255.0

    # Add an extra dimension to match the model's input shape
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
imgA = Input(shape=(64, 64,1))
imgB = Input(shape=(64, 64,1))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)
distance = Lambda(euclidean_distance)([featA, featB])
outputs = Dense(1, activation="sigmoid")(distance)

@app.route('/training',methods=['POST'])
def train():    
    getGambar1 = request.files['image1']
    getGambar2 = request.files['image2']    
    getGambar1.save(getGambar1.filename)
    getGambar2.save(getGambar2.filename)
    idUser = request.form.get("idUser")
    #Building Model
    model = Model(inputs=[imgA, imgB], outputs=outputs)
    model.load_weights("./transfer.h5")
    
    credentials_path = r"nyobaaja-973da4b3851c.json"
    client = storage.Client.from_service_account_json(credentials_path)
    bucket_name = 'upload_foto'
    bucket = client.get_bucket(bucket_name)


    #Import Gambar Person

    gambar1=preprocess_image(np.array(Image.open(getGambar1)))[0]#Gambar Lurus
    gambar2=preprocess_image(np.array(Image.open(getGambar2)))[0]#Gambar Samping
    gambar3=gambar2[:,::-1] #Balik Gambar
    #Import Gambar Lawan
    pic1=preprocess_image(cv2.imread("./lawan_1.png"))[0][:,:,0]
    pic2=preprocess_image(cv2.imread("./lawan_2.png"))[0][:,:,0]
    pic3=preprocess_image(cv2.imread("./lawan_3.png"))[0][:,:,0]
    
    
    
    gambar1Name = str(uuid.uuid4())+"_"+getGambar1.filename
    gambar2Name = str(uuid.uuid4())+"_"+getGambar2.filename

    
    blob1 = bucket.blob('fotoselfie/{}'.format(gambar1Name))
    blob2 = bucket.blob('fotoselfie/{}'.format(gambar2Name))
    blob1.upload_from_filename(getGambar1.filename)
    blob2.upload_from_filename(getGambar2.filename)
    linkFoto1 = blob1.public_url
    linkFoto2 = blob2.public_url
    
    os.remove(getGambar1.filename)
    os.remove(getGambar2.filename)
    
    #Array
    arr_gambar=np.array([gambar1,gambar2,gambar3])
    arr_pic=np.array([pic1,pic2,pic3])
    #arr_gambar=arr_gambar[:,:,0]
    #arr_pic=arr_pic[:,:,0]
    #Shuffling
    print(arr_gambar.shape, arr_pic.shape)
    gab=np.append(arr_gambar,arr_pic,axis=0)
    gablabel=np.append(np.ones(3),np.zeros(3))
    sampled_indices = np.random.choice(gab.shape[0], size=(3,), replace=False)
    sampled_array = gab[sampled_indices]
    sampled_label = gablabel[sampled_indices]
    unsampled_array=gab[~np.isin(np.arange(6), sampled_indices)]
    unsampled_label=gablabel[~np.isin(np.arange(6), sampled_indices)]
    image_data=[]
    label_data=[]
    for i in range(3):
        for j in range(3):
            image_data.append([sampled_array[i],unsampled_array[j]])
            label_data.append(sampled_label[i]*unsampled_label[j])
    label_data=np.array(label_data)
    image_data=np.array(image_data)
    
    #Building Model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  
    print(image_data.shape, label_data.shape)
    history = model.fit([image_data[:, 0], image_data[:, 1]], label_data[:],validation_split=0.1,batch_size=64,epochs=50)
    #Saving Model
    h5 = str(uuid.uuid4())+"_"+idUser+"_train.h5"
    
    
    model.save_weights("./"+h5)
    h5Blob = bucket.blob('thomasandfriend/{}'.format(h5))
    h5Blob.upload_from_filename("./"+h5)
    linkModel = h5Blob.public_url

    
    os.remove("./"+h5)
    
    return json.dumps({"linkFoto1": linkFoto1, "linkFoto2": linkFoto2, "linkmodel": linkModel})

@app.route('/predict', methods=['POST'])
def predict():
    print('masuk')

    getGambar1 = request.form.get('linkFoto1')
    getGambar1 = requests.get(getGambar1)
    getGambar2 = request.form.get('linkFoto2')
    getGambar2 = requests.get(getGambar2)
    getModel = request.form.get('linkModel')
    modelFileName = getModel.split('/')[-1]
    getModel = requests.get(getModel)
    
    
    
    with open('./'+modelFileName, 'wb') as f:
        f.write(getModel.content)
    
    

    #Building Model
    validasi= np.array(Image.open(BytesIO(getGambar1.content)))
    anchor= np.array(Image.open(BytesIO(getGambar2.content)))
    #Loading Weights
    new_model = Model(inputs=[imgA, imgB], outputs=outputs)
    # new_model.load_weights("./nathan_pasrah.h5")
    new_model.load_weights("./"+modelFileName)

    #processing
    validasi=preprocess_image(validasi)[0]
    anchor=preprocess_image(anchor)[0]
    print(validasi.shape,anchor.shape)
    #validasi=validasi[:,:]
    #anchor=anchor[:,:]

    #Prediction 
    Hasil=new_model.predict([np.array([validasi]),np.array([anchor])])[0][0]
    print(type(Hasil))
    
    os.remove("./"+modelFileName)

    #Threshold
    if Hasil>0.4:
        return json.dumps({'Predict':True})
    else:
        return json.dumps({'Predict':False})


if __name__ == '__main__':
    app.run('0.0.0.0', 8080, debug=True, threaded=True)