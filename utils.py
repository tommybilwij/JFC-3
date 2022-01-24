from typing import Text

import absl
import tensorflow as tf

from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs


_IMAGE_KEY = 'image_raw'
_LABEL_KEY = 'label'


#### INSERT GROUP 2's PREPROCESSING CODE ####
IMAGE_SIZE = 32


def _transformed_name(key):
    return key + '_xf'


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _image_parser(image_str):
    image = tf.image.decode_image(image_str, channels=3)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.cast(image, tf.float32) / 255.
    return image

def _label_parser(label_id):
    label = tf.one_hot(label_id, 10)
    return label


def preprocessing_fn(inputs):
    outputs = {_transformed_name(_IMAGE_KEY): tf.compat.v2.map_fn(_image_parser, tf.squeeze(inputs[_IMAGE_KEY], axis=1),
                                                                  dtype=tf.float32),
               _transformed_name(_LABEL_KEY): tf.compat.v2.map_fn(_label_parser, tf.squeeze(inputs[_LABEL_KEY], axis=1),
                                                                  dtype=tf.float32)
               }
    return outputs
############################################


def _input_fn(file_pattern: Text, tf_transform_output: tft.TFTransformOutput, batch_size: int) -> tf.data.Dataset:
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(_LABEL_KEY))
    return dataset


#### INSERT GROUP 2's MODEL CODE ####
def _build_keras_model() -> tf.keras.Model:
    inputs = keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3),name=_transformed_name(_IMAGE_KEY))
    d1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(inputs)
    d2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(d1)
    d3 = keras.layers.Flatten()(d2)
    outputs = keras.layers.Dense(10, activation='softmax')(d3)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')])
    
    model.summary(print_fn=absl.logging.info)
    
#     absl.logging.info(model.summary())
    return model
##########################################


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""
 
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
    train_batch_size = 32
    eval_batch_size = 32
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_input_fn = _input_fn(fn_args.train_files, tf_transform_output,batch_size=train_batch_size)

    eval_input_fn = _input_fn(fn_args.eval_files, tf_transform_output,batch_size=eval_batch_size)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model()
        
    model.fit(
        train_input_fn,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_input_fn,
        validation_steps=fn_args.eval_steps)
    
    
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)