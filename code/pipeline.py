"""Main pipeline file"""
from kubernetes import client as k8s_client
import kfp.dsl as dsl
import kfp.compiler as compiler

import tensorflow as tf

@dsl.pipeline(
  name='Construction LOR',
  description='Kubeflow Pipeline - SSL model'
)
def constructionlor(
    tenant_id,
    service_principal_id,
    service_principal_password,
    subscription_id,
    resource_group,
    workspace,
    EPOCHS,
):
  """Pipeline steps"""

  ## Pipeline parameters
  persistent_volume_path = '/mnt/azure' # storage disk to store datasets and models
  data_download = 'https://dl.dropboxusercontent.com/s/at3s44al3n3kymo/construction_dataset.zip?dl=0' # get the zip file from dropbox
  model_name = 'constructionlor'
  operations = {}
  model_folder = 'model'

  ## Group 2 parameters 
  AUTO = tf.data.AUTOTUNE   # Tuneable hyperperameters
  BATCH_SIZE = 128
  CROP_TO = 30 #the larger the image size, the longer the training time. But images that are too small will be too blurry to analyse. 
  SEED = 26
  INITIAL_LEARNING_RATE = 0.005
  PRETRAINING_MOMENTUM = 0.6
  LINEAR_MODEL_MOMENTUM = 0.9
  WEIGHT_DECAY = 0.0005

  # Others
  PROJECT_DIM = 2048
  LATENT_DIM = 512

  batch_size = 10

  # preprocess data
  operations['data collection'] = dsl.ContainerOp(
    name='data collection',
    image='kubeflowstorage.azurecr.io/data_collection:2',
    command=['python'],
    arguments=[
      '/scripts/data.py',
      '--base_path', persistent_volume_path,
      '--zfile', data_download
    ]
  )

  # preprocess and train
  operations['preprocess and training'] = dsl.ContainerOp(
    name='preprocess and training',
    image='kubeflowstorage.azurecr.io/preprocess_training:2',
    command=['python'],
    arguments=[
      '/scripts/train.py',
      '--base_path', persistent_volume_path,
      '--AUTO', AUTO,
      '--BATCH_SIZE', BATCH_SIZE,
      '--EPOCHS', EPOCHS,
      '--CROP_TO', CROP_TO,
      '--SEED', SEED,
      '--INITIAL_LEARNING_RATE', INITIAL_LEARNING_RATE,
      '--PRETRAINING_MOMENTUM', PRETRAINING_MOMENTUM,
      '--LINEAR_MODEL_MOMENTUM', LINEAR_MODEL_MOMENTUM,
      '--WEIGHT_DECAY', WEIGHT_DECAY,
      '--PROJECT_DIM', PROJECT_DIM,
      '--LATENT_DIM', LATENT_DIM,
      '--batch_size', batch_size,
      '--outputs', model_folder,
    ]
  )
  operations['preprocess and training'].after(operations['data collection'])

  # register model
  operations['register'] = dsl.ContainerOp(
    name='register',
    image='kubeflowstorage.azurecr.io/register:2',
    command=['python'],
    arguments=[
      '/scripts/register.py',
      '--base_path', persistent_volume_path,
      '--model', 'latest.h5',
      '--model_name', model_name,
      '--tenant_id', tenant_id,
      '--service_principal_id', service_principal_id,
      '--service_principal_password', service_principal_password,
      '--subscription_id', subscription_id,
      '--resource_group', resource_group,
      '--workspace', workspace
    ]
  )
  operations['register'].after(operations['preprocess and training'])

  operations['deploy'] = dsl.ContainerOp(
    name='deploy',
    image='kubeflowstorage.azurecr.io/deploy:3',
    command=['sh'],
    arguments=[
      '/scripts/deploy.sh',
      '-n', model_name,
      '-m', model_name,
      '-i', '/scripts/inferenceconfig.json',
      '-d', '/scripts/deploymentconfig.json',
      '-t', tenant_id,
      '-r', resource_group,
      '-w', workspace,
      '-s', service_principal_id,
      '-p', service_principal_password,
      '-u', subscription_id,
      '-b', persistent_volume_path
    ]
  )
  operations['deploy'].after(operations['register'])
  for _, op_1 in operations.items():
    op_1.container.set_image_pull_policy("Always")
    op_1.add_volume(
      k8s_client.V1Volume(
        name='azure',
        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
          claim_name='azure-managed-disk')
      )
    ).add_volume_mount(k8s_client.V1VolumeMount(
      mount_path='/mnt/azure', name='azure'))

if __name__ == '__main__':
  compiler.Compiler().compile(constructionlor, __file__ + '.tar.gz')
