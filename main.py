from io import BytesIO
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from google.cloud import storage
from google.cloud import secretmanager
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k
import numpy as np
import json
import uuid
import requests
import os
from dotenv import load_dotenv
import pymysql
import dlib

load_dotenv()  # Load environment variables from .env file
detector = dlib.get_frontal_face_detector()
tf.keras.backend.clear_session()
app = Flask(__name__)
CORS(app)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('accuracy')>0.8:
            print('\nAccuracy ngestop epoch')
            self.model.stop_training =True
            
callbacks = myCallback()

# # MySQL configurations
# app.config['MYSQL_HOST'] = 
# app.config['MYSQL_USER'] = 
# app.config['MYSQL_PASSWORD'] = 
# app.config['MYSQL_DB'] = 

# dbConn = pymysql.connect(
#     host=os.getenv('DB_HOST'),
#     user=os.getenv('DB_USER'),
#     password= os.getenv('DB_PASS'),
#     database=os.getenv('DB_NAME'),
#     cursorclass=pymysql.cursors.DictCursor
# )

def preprocess_image(image):
    desired_size = (64, 64)
    image = Image.fromarray(image)
    image = image.resize(desired_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def create_model():
    inputs = Input((64, 64, 1))
    x = Conv2D(64, (11, 11), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    pooledOutput = Dense(2048)(pooledOutput)
    outputs = Dense(128)(pooledOutput)

    model = Model(inputs, outputs)
    return model

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))


def get_faces(picture):
    # gray=picture.mean(axis=2)
    faces = detector(picture)
    if len(faces) == 1:
        for face in faces:
            # Get the coordinates of the face
            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()

            # Draw a rectangle around the face
            Crop=picture[int(y+0.05*h):int(y+0.95*h),int(x+0.05*w):int(x+0.95*w)]
        return Crop

feature_extractor = create_model()
imgA = Input(shape=(64, 64, 1))
imgB = Input(shape=(64, 64, 1))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)
distance = Lambda(euclidean_distance)([featA, featB])
outputs = Dense(1, activation="sigmoid")(distance)

@app.route('/training', methods=['POST'])
def train():
        dbConn = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password= os.getenv('DB_PASS'),
            database=os.getenv('DB_NAME'),
            cursorclass=pymysql.cursors.DictCursor
        )
        getGambar1 = request.files['image1']
        getGambar2 = request.files['image2']    
        getGambar1.save(getGambar1.filename)
        getGambar2.save(getGambar2.filename)
        idUser = request.form.get("idUser")
        sql = dbConn.cursor()
        
        query1 = "SELECT id FROM user WHERE id = %s"
        
        sql.execute(query1, (f"{idUser}"))
        data = sql.fetchall()
        
        # if len(data) == 0:
        #     return json.dumps({'error': 'true', 'message': 'Data tidak terdaftar!'})
        
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        model.load_weights("./transfer_sample1.h5")
        
        gcpClient = secretmanager.SecretManagerServiceClient()
        
        keysName = f"projects/872765504345/secrets/gcs-key/versions/latest"
        
        response = gcpClient.access_secret_version(request={"name": keysName})
        
        credentials = json.loads(response.payload.data.decode('UTF-8'))
        
        client = storage.Client.from_service_account_info(credentials)
        bucket_name = 'suara-kita'
        bucket = client.get_bucket(bucket_name)


        gambar1=np.array(Image.open(getGambar1))
        print(gambar1.shape)#Gambar Lurus
        gambar1=get_faces(gambar1)#udah dicrop, blom gray
        gambar1=preprocess_image(gambar1)[0]
        gambar1=np.mean(gambar1,axis=2)#Grayscale
        gambar2=np.array(Image.open(getGambar2))#Gambar Samping
        gambar2=get_faces(gambar2)#udah dicrop, blom gray
        gambar2=preprocess_image(gambar2)[0]
        gambar2=np.mean(gambar2,axis=2)#Grayscale


        gambar3=gambar2[:,::-1]

        pic1=preprocess_image(np.array(tf.keras.preprocessing.image.load_img("./lawan_1.png")))[0]
        pic2=preprocess_image(np.array(tf.keras.preprocessing.image.load_img("./lawan_2.png")))[0]
        pic3=preprocess_image(np.array(tf.keras.preprocessing.image.load_img("./lawan_3.png")))[0]

        pic1=np.mean(pic1,axis=2)
        pic2=np.mean(pic2,axis=2)
        pic3=np.mean(pic3,axis=2)

        print(gambar1.shape,gambar2.shape,gambar3.shape,pic1.shape,pic2.shape,pic3.shape)
        
        
        gambar1Name = str(uuid.uuid4())+"_"+getGambar1.filename
        gambar2Name = str(uuid.uuid4())+"_"+getGambar2.filename

        
        blob1 = bucket.blob('fotoselfie/{}'.format(gambar1Name))
        blob2 = bucket.blob('fotoselfie/{}'.format(gambar2Name))
        blob1.upload_from_filename(getGambar1.filename)
        blob2.upload_from_filename(getGambar2.filename)
        linkFoto1 = blob1.public_url #
        linkFoto2 = blob2.public_url #
        
        os.remove(getGambar1.filename)
        os.remove(getGambar2.filename)
        
        #Array
        arr_gambar=np.array([gambar1,gambar2,gambar3])
        arr_pic=np.array([pic1,pic2,pic3])
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
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])  
        print(image_data.shape, label_data.shape)
        history = model.fit([image_data[:, 0], image_data[:, 1]], label_data[:],validation_split=0.3,batch_size=64,epochs=20,callbacks=[callbacks])
        #Saving Model
        h5 = str(uuid.uuid4())+"_model"+idUser+"_train.h5"
        
        
        
        model.save_weights("./"+h5.strip())
        h5Blob = bucket.blob('train-model/{}'.format(h5))
        h5Blob.upload_from_filename("./"+h5)
        linkModel = h5Blob.public_url #
        
        query2 = "INSERT INTO fotouser(idUser, listFoto1, listFoto2, model) VALUES (%s, %s, %s, %s)"
        values = (f"{idUser}", linkFoto1, linkFoto2, linkModel)
        sql.execute(query2, values)
        
        dbConn.commit()
        
        os.remove(h5)        
        sql.close()
        dbConn.close()
        tf.keras.backend.clear_session()
        return json.dumps({
            "error": "false",
            "message": "Data berhasil diinput",
        })




@app.route('/predict', methods=['POST'])
def predict():
    dbConn = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password= os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME'),
        cursorclass=pymysql.cursors.DictCursor
    )
    getGambar1 = request.files['Gambar']#Files Gambar Aldo
    getGambar1=np.array(Image.open(getGambar1))
    getGambar1=get_faces(getGambar1)#Crop
    getGambar1=preprocess_image(getGambar1)[0]
    getGambar1=np.mean(getGambar1,axis=2)#Grayscale

    getIdUser = request.form.get('idUser')
    sql = dbConn.cursor()

    query1 = "SELECT listFoto1, listFoto2, model FROM fotouser WHERE idUser = %s LIMIT 1" 
    sql.execute(query1, (f"{getIdUser}"))
    data = sql.fetchall()
    
    # if len(data) == 0:
    #     return json.dumps({'error': 'true', 'message': 'Data tidak terdaftar!'})

    getGambar2 = data[0][0]
    getModel = data[0][2]
    
    # getGambar2 = request.form.get('linkFoto2')
    getGambar2 = requests.get(getGambar2)
    # getModel = request.form.get('linkModel')
    modelFileName = getModel.split('/')[-1]
    getModel = requests.get(getModel)
    
    with open('./'+modelFileName, 'wb') as f:
        f.write(getModel.content)
    
    #Building Model
    validasi= getGambar1
    anchor= np.array(Image.open(BytesIO(getGambar2.content)))
    new_model = Model(inputs=[imgA, imgB], outputs=outputs)
    new_model.load_weights("./"+modelFileName)

    #processing
    # validasi=preprocess_image(validasi)[0]
    anchor=get_faces(anchor)
    anchor=preprocess_image(anchor)[0]
    anchor=np.mean(anchor,axis=2)
    print(validasi.shape,anchor.shape)

    #Prediction 
    Hasil=new_model.predict([np.array([validasi]),np.array([anchor])])[0][0]
    print(Hasil, type(Hasil))

    os.remove("./"+modelFileName)

    #Threshold
    if Hasil>0.4:
        # sql.execute("UPDATE user SET verified = 1 WHERE id = %s", (getIdUser,))
        
        # dbConn.commit()
        # sql.close()
        #dbConn.close()
        tf.keras.backend.clear_session()
        
        
        return json.dumps({'error': 'false', 'message': 'Data tervalidasi','hasilPredict':'true'})
    else:    
        # sql.close()
        # dbConn.close()
        tf.keras.backend.clear_session()
        #dbConn.close()
        
        return json.dumps({'error': 'true', 'message': 'Data tidak valid!', 'hasilPredict':'false'})


if __name__ == '__main__':
    app.run(debug=True)
