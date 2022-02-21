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

import json


def init():
    """ Main function to load the model
    """
    global model
    if Model.get_model_path('constructionlors'):
        model_path = Model.get_model_path('constructionlors')
    else:
        model_path = 'mnt/azure/model/latest.h5'
    # model_path = '/Users/tommybillywijaya/Documents/_Main_Document/Projects/New/work/azurepipeline_live/data/model/latest.h5'


    print('Attempting to load model')
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print('Done!')

    print('Initialized model "{}" at {}'.format(
        model_path, datetime.datetime.now()))


def run(raw_data):
    """ to predict based on the trained model
        raw_data     : the json inputs
        list_classes : list of 13 categories 
    """

    # list all categories
    list_classes=['aerial shots', 'concrete', 'scaffolding', 'indoor furnishings', 'rebar', 'road', 'railway', 'fire damage', 'columns and beams', 'workers', 'plumbing', 'electrical details', 'machinery']

    prev_time = time.time()

    post = json.loads(str(raw_data))
    print(post)
    img_path = post['image']

    current_time = time.time()

    tensor = process_image(img_path, 30)
    t = tf.reshape(tensor, [-1, 30, 30, 3])

    ## Define model prediction
    o = model.predict(t, steps=1)  # [0][0]

    # get all predicted scores
    score = []
    for i in range(0,13):
        score.append(o[0][i])

    print(o)
    class_num=np.argmax(o, axis=1)[0]
    predicted_class=list_classes[class_num]
    print(predicted_class)

    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    # float32 is not serialisable, so it must be converted to float64
    payload = {
        'time': inference_time.total_seconds(),
        'prediction': predicted_class,
        'confidence level': np.float64(score[class_num]),
        list_classes[0]+'_scores': np.float64(score[0]),
        list_classes[1]+'_scores': np.float64(score[1]),
        list_classes[2]+'_scores': np.float64(score[2]),
        list_classes[3]+'_scores': np.float64(score[3]),
        list_classes[4]+'_scores': np.float64(score[4]),
        list_classes[5]+'_scores': np.float64(score[5]),
        list_classes[6]+'_scores': np.float64(score[6]),
        list_classes[7]+'_scores': np.float64(score[7]),
        list_classes[8]+'_scores': np.float64(score[8]),
        list_classes[9]+'_scores': np.float64(score[9]),
        list_classes[10]+'_scores': np.float64(score[10]),
        list_classes[11]+'_scores': np.float64(score[11]),
        list_classes[12]+'_scores': np.float64(score[12]),
    }



    print('Input ({}), Prediction ({})'.format(post['image'], payload))

    return payload


def process_image(path, image_size):
    """ Extract image (from web or path)
        path       : path directory to the image
        image_size : size of the image
    """
    if path.startswith('http'):
        response = requests.get(path)
        img = np.array(Image.open(BytesIO(response.content)))
    else:
        img = np.array(Image.open(path))

    img_tensor = tf.convert_to_tensor(img, dtype=tf.float64)
    # tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    return img_final


def info(msg, char="#", width=75):
    # print to console in a beautiful way
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1 * width) + 5, msg) + char)
    print(char * width)


if __name__ == "__main__":

    # Create a dictionary of URL images
    images = {
        'concrete': 'https://www.techexplorist.com/wp-content/uploads/2017/07/concrete.jpg',  # noqa: E501
        'railway': 'http://www.railway-fasteners.com/uploads/allimg/how-to-build-a-railway-track.jpg'  # noqa: E501
    }

    # Run the main function
    init()

    for k, v in images.items():
        print('{} => {}'.format(k, v))

    info('Concrete Test')
    concrete= json.dumps({'image': images['concrete']})
    print(concrete)
    run(concrete)

    info('Railway Test')
    railway = json.dumps({'image': images['railway']})
    print(railway)
    run(railway)
