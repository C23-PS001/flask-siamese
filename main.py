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
import matplotlib.pyplot as plt
import cv2
import numpy as np


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
def rizal():
    print('masuk')
    #Building Model
    model = Model(inputs=[imgA, imgB], outputs=outputs)
    model.load_weight(r"C:\Users\natha\Bangkit\ForCapstone\transfer.h5")

    #Import Gambar Person
    gambar1=""#Gambar Lurus
    gambar2=""#Gambar Samping
    gambar3=gambar2[:,::-1,:]

    #Import Gambar Lawan
    pic1=""
    pic2=""
    pic3=""

    #Array
    arr_gambar=""
    arr_pic=""
    arr_gambar=arr_gambar[:,:,0]
    arr_pic=arr_pic[:,:,0]

    #Shuffling
    gab=np.append(arr_gambar,arr_pic,axis=0)
    gablabel=np.append(np.ones(10),np.zeros(10))
    sampled_indices = np.random.choice(gab.shape[0], size=(10,), replace=False)
    sampled_array = gab[sampled_indices]
    sampled_label = gablabel[sampled_indices]
    unsampled_array=gab[~np.isin(np.arange(20), sampled_indices)]
    unsampled_label=gablabel[~np.isin(np.arange(20), sampled_indices)]
    image_data=[]
    label_data=[]
    for i in range(10):
        for j in range(10):
            image_data.append([sampled_array[i],unsampled_array[j]])
            label_data.append(sampled_label[i]*unsampled_label[j])
    label_data=np.array(label_data)
    image_data=np.array(image_data)
    
    #Building Model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])    
    history = model.fit([image_data[:, 0], image_data[:, 1]], label_data[:],validation_split=0.1,batch_size=64,epochs=50)

    #Saving Model
    model.save_weights(r"C:\Users\natha\Downloads\nathan pasrah.h5")



    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file)        
        results = reader.readtext(image)
        List= []
        for result in results:
            List.append(result[1])
        #print(image_list[:20])
        # TTL_SPLIT= CariTTL(List)
        return json.dumps({'NIK':CariNIK(List),'Nama':CariNama(List),'Tgl Lahir':CariTTL(List)})

    return 'No image file found'

@app.route('/predict', methods=['POST'])
def Udin(validasi, anchor):
    print('masuk')

    #Building Model
    
    #Loading Weights
    new_model = Model(inputs=[imgA, imgB], outputs=outputs)
    new_model.load_weights(r"C:\Users\natha\Downloads\nathan pasrah.h5")

    #processing
    validasi=preprocess_image(validasi)[0]
    anchor=preprocess_image(anchor)[0]
    validasi=validasi[:,:,0]
    anchor=anchor[:,:,0]

    #Prediction 
    Hasil=new_model.predict([np.array([validasi]),np.array([anchor])])[0][0]

    #Threshold
    if Hasil>0.4:
        return True
    else:
        return False


if __name__ == '__main__':
    app.run('0.0.0.0', 1891, debug=True, threaded=True)