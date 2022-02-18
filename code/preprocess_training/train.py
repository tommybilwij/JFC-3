from __future__ import absolute_import, division, print_function
import os
import math
import hmac
import json
import hashlib
import argparse
from random import shuffle
from pathlib2 import Path
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, random_rotation
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add 


##--------------------------------- GROUP 2's code START ---------------------------------
## Defining our data augmentation pipeline
# Preprocessing:
# 1.) Random flip
# 2.) Random rotate
# 3.) Color distortions: random contrast, random saturaiton, random hue, rgb to greyscale
def flip_random(image):
    # With random crops we also apply horizontal flipping.
    image = tf.image.random_flip_left_right(image)
    return image

def rotate_random(x):
    x = tf.image.rot90(x, k=1)
    return x

def color_jitter(x):
    x = tf.image.random_brightness(x, max_delta=0.5)
    x = tf.image.random_contrast(
        x, lower= 0.7, upper=1.7
    )
    x = tf.image.random_saturation(
        x, lower=0.7, upper=1.6
    )

    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x

def color_hue(x):
    x = tf.image.random_hue(x, max_delta=0.04)
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random(image)
    image = random_apply(color_jitter, image, p=0.9)
    image = random_apply(color_hue, image, p=0.8)
    image = random_apply(rotate_random, image, p=0.1)
    image = random_apply(color_drop, image, p=0.1)
    return image

## Defining the encoder and the predictor
# We use an implementation of ResNet50. The code is taken from the keras-idiomatic-programmer repository. 
# The code has been downloaded and saved as simsiam_functions.py
def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
    x = ZeroPadding2D(padding=(3, 3))(inputs)
    
    # First Convolutional layer uses large (coarse) filter
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x

def identity_block(x, n_filters):
    """ Construct a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
    """
    
    # Save input vector (feature maps) for the identity link
    shortcut = x
    
    ## Construct the 1x1, 3x3, 1x1 convolution block
    
    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([shortcut, x])
    return x


def projection_block(x, n_filters, strides=(2,2)):
    """ Construct a Bottleneck Residual Block of Convolutions with Projection Shortcut
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
        strides  : whether the first convolution is strided
    """
    # Construct the projection shortcut
    # Increase filters by 4X to match shape when added to output of block
    shortcut = BatchNormalization()(x)
    shortcut = Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(shortcut)

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck layer
    # Feature pooling when strides=(2, 2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of filters by 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([x, shortcut])
    return x


def group(x, n_filters, n_blocks, strides=(2, 2)):
    """ Construct a Residual Group
        x         : input into the group
        n_filters : number of filters for the group
        n_blocks  : number of residual blocks with identity link
        strides   : whether the projection block is a strided convolution
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters)
    return x


def learner(x, groups):
    """ Construct the Learner
        x     : input to the learner
        groups: list of groups: number of filters and blocks
    """
    # First Residual Block Group (not strided)
    n_filters, n_blocks = groups.pop(0)
    x = group(x, n_filters, n_blocks, strides=(1, 1))
    print(x)
    # Remaining Residual Block Groups (strided)
    for n_filters, n_blocks in groups:
        x = group(x, n_filters, n_blocks)
    return x

def get_encoder(CROP_TO,PROJECT_DIM,WEIGHT_DECAY):
    # Meta-parameter: list of groups: number of filters and number of blocks
    groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],           # ResNet50
            }

    # Input and backbone.
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(
        inputs
    )
    x = stem(x)
    x = learner(x, groups[50]) 
    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    # Projection head.
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    outputs = layers.BatchNormalization()(x)
    return tf.keras.Model(inputs, outputs, name="encoder")


def get_predictor(PROJECT_DIM,LATENT_DIM,WEIGHT_DECAY):
    model = tf.keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            layers.Input((PROJECT_DIM,)),
            layers.Dense(
                LATENT_DIM,
                use_bias=False,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            ),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dense(PROJECT_DIM),
        ],
        name="predictor",
    )
    return model


## Defining the (pre-)training loop
# One of the main reasons behind training networks with these kinds of approaches 
# is to utilize the learned representations for downstream tasks like classification.
# We start by defining the loss function.
def compute_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

class SimSiam(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
##--------------------------------- GROUP 2's code END ---------------------------------


def info(msg, char="#", width=75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1 * width) + 5, msg) + char)
    print(char * width)

# Check whether target directory exists
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return Path(path).resolve(strict=False)


def generate_hash(dfile, key):
    print('Generating hash for {}'.format(dfile))
    m = hmac.new(str.encode(key), digestmod=hashlib.sha256)
    BUF_SIZE = 65536
    with open(str(dfile), 'rb') as myfile:
        while True:
            data = myfile.read(BUF_SIZE)
            if not data:
                break
            m.update(data)

    return m.hexdigest()


# @print_info
def run(
        CROP_TO,
        nonlabeled_input_folder,
        batch_size,
        AUTO,
        BATCH_SIZE,
        SEED,
        EPOCHS,
        INITIAL_LEARNING_RATE,
        PRETRAINING_MOMENTUM,
        LINEAR_MODEL_MOMENTUM,
        WEIGHT_DECAY,
        PROJECT_DIM,
        LATENT_DIM,
        output,):

    
    ##--------------------------------- GROUP 2's code START ---------------------------------

    # Initialise image size
    img_height, img_width = (CROP_TO, CROP_TO)
    
    AUTO= tf.data.AUTOTUNE
    
    # Load and format the images from the nonlabeled_input_dataset folder into numpy arrays
    number_of_images = len(os.listdir(nonlabeled_input_folder)) #count the number of files inside the nonlabeled_input_dataset folder
    x_train_nonlabeled = np.empty([number_of_images, img_height, img_width, 3]) #create an empty array that will store all the image arrays. 

    x = 0
    for i in os.listdir(nonlabeled_input_folder): #look into all the files inside the nonlabeled_input_dataset folder
        img = image.load_img(nonlabeled_input_folder + "/" + i, target_size = (img_height,img_width)) #load all the images within the set image size
        #convert the images into a nunmpy array
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        #combine all the image arrays into one big array called x_train_nonlabeled
        x_train_nonlabeled[x] = input_arr
        x +=1

    x_train_nonlabeled.shape

    # Reduce datasize to 5000 to reduce training time
    random_int_array = np.random.randint(0, len(x_train_nonlabeled), int(len(x_train_nonlabeled)*2/3))
    x_train_nonlabeled = np.delete(x_train_nonlabeled, random_int_array, axis=0)
    x_train_nonlabeled.shape

    # Load and format the images from the labeled_input_dataset folder into numpy arrays with labels
    train_data_dir = str(base_path)+"/processed_data/train"
    test_data_dir = str(base_path)+"/processed_data/test"
    train_datagen = ImageDataGenerator(rotation_range = 10, horizontal_flip=True)


    # Loads the image into an array. 
    # The labels are determined by the folder names. 
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = train_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical')

    #store all the processed image arrays into one array, same for the labels. 
    x_train_labeled =np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
    y_train_labeled =np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])

    x_test_labeled = np.concatenate([test_generator.next()[0] for i in range(test_generator.__len__())])
    y_test_labeled =np.concatenate([test_generator.next()[1] for i in range(test_generator.__len__())])

    # Convert the data into TensorFlow Dataset objects
    ## Here we create two different versions of our dataset without any ground-truth labels using the nonlabeled_input_dataset. 
    ## The outputs from both datasets are shown.
    #first dataset
    ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train_nonlabeled)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=SEED)
        .map(custom_augment, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    #second dataset
    ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train_nonlabeled)
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=SEED)
        .map(custom_augment, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))  

    ## Pre-training our networks
    # Training the unsupervised learning network using nonlabeled_input_dataset.
    # Will output a cosine similarity decay graph that shows the ability of the model to match images from the two datasets.
    # Create a cosine decay learning scheduler.
    num_training_samples = len(x_train_nonlabeled)
    steps = EPOCHS * (num_training_samples // BATCH_SIZE)
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE, decay_steps=steps
    )

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )

    # Compile model and start training.
    simsiam = SimSiam(get_encoder(CROP_TO,PROJECT_DIM,WEIGHT_DECAY), get_predictor(PROJECT_DIM,LATENT_DIM,WEIGHT_DECAY))
    simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=PRETRAINING_MOMENTUM))
    history = simsiam.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])

    ## Evaluating our SSL method
    # The most popularly used method to evaluate a SSL method in computer vision (or any other pre-training method as such) 
    # is to learn a linear classifier on the frozen features of the trained backbone model (in this case it is ResNet50) 
    # and evaluate the classifier on unseen images.
    # Trained using the labeled_input_dataset.
    # We first create labeled `Dataset` objects.

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_labeled, y_train_labeled))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test_labeled, y_test_labeled))

    # Then we shuffle, batch, and prefetch this dataset for performance. We
    # also apply random flip as an augmentation but only to the
    # training set.
    train_ds = (
        train_ds.shuffle(1024)
        .map(lambda x, y: (flip_random(x), y), num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

    # Extract the backbone ResNet50.
    backbone = tf.keras.Model(
        simsiam.encoder.input, simsiam.encoder.get_layer("backbone_pool").output
    )

    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = backbone(inputs, training=False)
    outputs = layers.Dense(train_generator.num_classes, activation="softmax")(x)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")

    # Compile model and start training.
    linear_model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=LINEAR_MODEL_MOMENTUM),
    )

    history = linear_model.fit(
        train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[early_stopping]
    )
    _, test_acc = linear_model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))

    ##--------------------------------- GROUP 2's code END ---------------------------------


    # save model
    info('Saving Model')

    # check existence of base model folder
    output = check_dir(output)

    print('Serializing into saved_model format')
    tf.saved_model.save(linear_model, str(output))
    print('Done!')

    # add time prefix folder
    file_output = str(Path(output).joinpath('latest.h5'))
    print('Serializing h5 model to:\n{}'.format(file_output))
    linear_model.save(file_output)

    return generate_hash(file_output, 'kf_pipeline')


def generate_hash(dfile, key):
    print('Generating hash for {}'.format(dfile))
    m = hmac.new(str.encode(key), digestmod=hashlib.sha256)
    BUF_SIZE = 65536
    with open(str(dfile), 'rb') as myfile:
        while True:
            data = myfile.read(BUF_SIZE)
            if not data:
                break
            m.update(data)

    return m.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='transfer learning for binary image task')
    parser.add_argument('-s', '--base_path',
                        help='directory to base data', default='../../data')
    parser.add_argument('-a', '--AUTO',
                        help='tuneable parameter', default=tf.data.AUTOTUNE)
    parser.add_argument('-b', '--BATCH_SIZE',
                        help='set BATCH_SIZE', default=128, type=int)
    parser.add_argument('-c', '--EPOCHS',
                        help='number of EPOCHS', default=5, type=int)
    parser.add_argument('-e', '--CROP_TO',
                        help='image size', default=30, type=int)
    parser.add_argument('-f', '--SEED',
                        help='initialise seed status', default=26, type=int)
    parser.add_argument('-q', '--INITIAL_LEARNING_RATE',
                        help='learning rate', default=0.005, type=float)
    parser.add_argument('-i', '--PRETRAINING_MOMENTUM',
                        help='pretraining momentum', default=0.6, type=float)
    parser.add_argument('-j', '--LINEAR_MODEL_MOMENTUM',
                        help='linear model momentum', default=0.9, type=float)
    parser.add_argument('-k', '--WEIGHT_DECAY',
                        help='weight decay', default=0.0005, type=float)
    parser.add_argument('-l', '--PROJECT_DIM',
                        help='project dimension', default=2048, type=int)
    parser.add_argument('-m', '--LATENT_DIM',
                        help='latent dimension', default=512, type=int)
    parser.add_argument('-y', '--batch_size',
                        help='set batch_size', default=10, type=int)
    parser.add_argument('-o', '--outputs',
                        help='output directory', default='model')
    parser.add_argument('-ff', '--dataset', help='cleaned data listing',default='train.txt')
    args = parser.parse_args()

    info('Using TensorFlow v.{}'.format(tf.__version__))

    # Initialise path for dataset
    base_path = Path(args.base_path).resolve(strict=False)

    nonlabeled_input_folder = str(base_path)+'/new_nonlabelled' #for unsupervised learning, is stored in /mnt/azure/data/

    params = Path(args.base_path).joinpath('params.json')

    target_path = Path(args.base_path).resolve(
        strict=False).joinpath(args.outputs)

    args = {
        "CROP_TO": args.CROP_TO,
        "nonlabeled_input_folder": nonlabeled_input_folder,
        "batch_size": args.batch_size,
        "AUTO": tf.data.AUTOTUNE,
        "BATCH_SIZE": args.BATCH_SIZE,
        "SEED": args.SEED,
        "EPOCHS": args.EPOCHS,
        "INITIAL_LEARNING_RATE": args.INITIAL_LEARNING_RATE,
        "PRETRAINING_MOMENTUM": args.PRETRAINING_MOMENTUM,
        "LINEAR_MODEL_MOMENTUM": args.LINEAR_MODEL_MOMENTUM,
        "WEIGHT_DECAY": args.WEIGHT_DECAY,
        "PROJECT_DIM": args.PROJECT_DIM,
        "LATENT_DIM": args.LATENT_DIM,
        "output": str(target_path),
    }

    # printing out args for posterity
    for i in args:
        print('{} => {}'.format(i, args[i]))

    model_signature = run(**args)

    args['dataset_signature'] = ''
    args['model_signature'] = model_signature.upper()
    args['model_type'] = 'tfv2-MobileNetV2'
    print('Writing out params...', end='')
    with open(str(params), 'w') as f:
        json.dump(args, f)

    print(' Saved to {}'.format(str(params)))

    print('Preprocess and Training component is complete')

