import json
import time
from io import BytesIO
import datetime
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

from azureml.core.model import Model

import os
import argparse
from pathlib2 import Path


def init():
    global model
    print('this here')
    if Model.get_model_path('constructionlor'):
        model_path = Model.get_model_path('constructionlor')
    else:
        model_path = str(base_path)+'/model/latest.h5'


    print('Attempting to load model')
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print('Done!')

    print('Initialized model "{}" at {}'.format(
        model_path, datetime.datetime.now()))


def run(raw_data,list_classes):
    prev_time = time.time()

    post = json.loads(raw_data)
    img_path = post['image']

    current_time = time.time()

    tensor = process_image(img_path, 30)
    t = tf.reshape(tensor, [-1, 30, 30, 3])

    ## Define model prediction
    o = model.predict(t, steps=1)  # [0][0]
    print(o)
    class_num=np.argmax(o, axis=1)[0]
    predicted_class=list_classes[class_num]
    print(predicted_class)

    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    payload = {
        'time': inference_time.total_seconds(),
        'prediction': predicted_class,
        list_classes[0]+'_scores': o[0][0],
        list_classes[1]+'_scores': o[0][1],
        list_classes[2]+'_scores': o[0][2],
        list_classes[3]+'_scores': o[0][3],
        list_classes[4]+'_scores': o[0][4],
        list_classes[5]+'_scores': o[0][5],
        list_classes[6]+'_scores': o[0][6],
        list_classes[7]+'_scores': o[0][7],
        list_classes[8]+'_scores': o[0][8],
        list_classes[9]+'_scores': o[0][9],
        list_classes[10]+'_scores': o[0][10],
        list_classes[11]+'_scores': o[0][11],
        list_classes[12]+'_scores': o[0][12],
    }

    print('Input ({}), Prediction ({})'.format(post['image'], payload))

    return payload


def process_image(path, image_size):
    # Extract image (from web or path)
    if path.startswith('http'):
        response = requests.get(path)
        img = np.array(Image.open(BytesIO(response.content)))
    else:
        img = np.array(Image.open(path))

    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    # tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    return img_final


def info(msg, char="#", width=75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1 * width) + 5, msg) + char)
    print(char * width)


if __name__ == "__main__":
    
    # Initialise path for dataset
    global base_path
    base_path = '/mnt/azure'
    base_path = Path(base_path).resolve(strict=False)

    print('here')
    folder = str(base_path)+"/new_labelled"
    list_classes = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    print(list_classes)

    images = {
        'concrete': 'https://www.techexplorist.com/wp-content/uploads/2017/07/concrete.jpg',  # noqa: E501
        'railway': 'http://www.railway-fasteners.com/uploads/allimg/how-to-build-a-railway-track.jpg'  # noqa: E501
    }

    init()

    for k, v in images.items():
        print('{} => {}'.format(k, v))

    info('Concrete Test')
    concrete= json.dumps({'image': images['concrete']})
    print(concrete)
    run(concrete,list_classes)

    info('Railway Test')
    railway = json.dumps({'image': images['railway']})
    print(railway)
    run(railway,list_classes)
