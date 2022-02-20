# NOTE: PLEASE REFER TO THE FINAL REPORT OF GROUP 3 TO UNDERSTAND THE THEORY BEHIND THE PIPELINE.

# **File structure**
Each directory contains a minimum of 4 files: 
- Build.sh
- DockerFile
- Requirements.txt
- A Python file (.py)

Build.sh and Dockerfile contain client code. Build.sh acquires the directory and creates a Docker Image (DI) for the component. This is the containerised file that is pushed onto Kubeflow. Meanwhile, the "Dockerfile" file includes the command that the container needs to execute to perform its function. 

Requirements.txt contains the custom Python packages required to run the component. This will be downloaded via PIP.

The Python file contains the functionalities of each component, thus making up the majority of the runtime code for each component. Each python file is what is described above in section 3.1.

## Special files in Deploy component
Deploy contains extra files. The extra files are:
- Environment.yml
- Deploy.sh & Deploymentconfig.json 
- Inference.sh & Inferenceconfig.json 

Environment.yml specifies the platforms used to create the pipeline, such as the Python version.

Deploy.sh is the primary file responsible for deploying the model onto the pipeline. It requires component 3.1.3: Register to grant authorisation to the developer to deploy the model. It also overwrites previous versions of the model if there are any.

Deploymentconfig.json determines the resource allocation to the model (No. of CPUs, memory allocation), encoding and enabling diagnostics.

Inference.sh tests the model with test image files specified in the score.py file to ensure the component, and hence the pipeline, functions from end-to-end.

Inferenceconfig.json outlines the requirements for this inference test, specifying which Python (score.py) and environment (environment.yml) file to run.


# **Using the Pipeline**

## **Test each component locally**

Data collection component <br />
*python3 data.py*

Preprocess and training component  <br />
*python3 train.py*

Register component  <br />
*python register.py --model_path v --model_name c --tenant_id c --service_principal_id v --service_principal_password v --subscription_id v --resource_group x --workspace c*

Deploy component  <br />
*Go to debugging section*

## **Publish containerised docker image to Container Registry**

Login into AZ account and subscription
*az login* <br />
*az account set --subscription <NAME OR ID OF SUBSCRIPTION>* <br />
Subscription ID: 03363605-92a3-4526-9b82-a1dcbe5983bc
  
Get the AKS cluster credential <br />
*az aks get-credentials -n <NAME> -g <RESOURCE_GROUP_NAME>* <br />
Name: clusterlor || Resource group: resourcetfx
  
Set the path in Container Registry that you want to push the containers to:  <br />
*export REGISTRY_PATH=<REGISTRY_NAME>.azurecr.io
Registry name: kubeflowstorage*

Run the following command to authenticate your Container Registry:  <br />
*az acr login --name <REGISTRY_NAME>*
Registry name: kubeflowstorage

Create a version, to be associated with your model each time it runs (change this accordingly):  <br />
*export VERSION_TAG=3*

Each docker image will be built and uploaded to the cloud using the Container Registry.
Run these commands to build images, and push them to Azure’ Container registry (make sure your local computer has enough storage)  <br />
*cd data_collection  <br />
docker build . -t ${REGISTRY_PATH}/data_collection:${VERSION_TAG}  <br />
docker push ${REGISTRY_PATH}/data_collection:${VERSION_TAG}*

*cd ../preprocess_training  <br />
docker build . -t ${REGISTRY_PATH}/preprocess_training:${VERSION_TAG}  <br />
docker push ${REGISTRY_PATH}/preprocess_training:${VERSION_TAG}*

*cd ../register  <br />
docker build . -t ${REGISTRY_PATH}/register:${VERSION_TAG}  <br />
docker push ${REGISTRY_PATH}/register:${VERSION_TAG}*

*cd ../deploy  <br /> 
docker build . -t ${REGISTRY_PATH}/deploy:${VERSION_TAG} <br />
docker push ${REGISTRY_PATH}/deploy:${VERSION_TAG}*

## **Upload pipeline.py.tar.gz to Kubeflow Dashboard**

Access Kubeflow dashboard  <br />
*kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80*  <br />
Go to http://localhost:8080 

pipeline.py must be executed in order to create pipeline.py.tar.gz, a file required by Kubernetes. This scoring URL can be found once we publish the pipeline to AKS using the Kubeflow dashboard and successfully execute the experiment.

http://b38f2a0b-8708-4e39-8539-cb57691ef197.eastus.azurecontainer.io/score?image=<Image_URL> 

Then replace “<Image_URL>” with the image you wish to classify.

## **Debugging method if deployment fails**

During deployment process, the log dashboard will appear in Kubeflow Dashboard for that experiment run:
<img width="966" alt="Screen Shot 2022-02-21 at 10 04 38 am" src="https://user-images.githubusercontent.com/53250006/154868325-983fbf64-7254-4b5b-a969-4b3504935fbd.png">

If error appears, deploy container can be debugged locally:
*az ml model deploy -n <model_name> -m <model_name>:<model_version> --ic inferenceconfig.json --dc deploymentconfig.json --resource-group resourcetfx --workspace-name tfxkubeflow --overwrite -v*

Check model version in AZ ML Workspace:
<img width="1052" alt="Screen Shot 2022-02-21 at 10 13 02 am" src="https://user-images.githubusercontent.com/53250006/154868600-855bc6cf-67b9-4309-8937-c3b841346eb8.png">

## **Run experiment**

Create an experiment, and start a run in Kubeflow dashboard for that pipeline. Fill all the parameters (as specified in pipeline.py).
![Screen Shot 2022-02-20 at 9 49 18 pm](https://user-images.githubusercontent.com/53250006/154868633-a749f8a7-ed1f-493f-9e45-eb55c28470f1.png)
  
Outputs can be seen in AZ ML Workspace (Endpoint to see the container instances created, Model to see all the deployed models)
<img width="1053" alt="Screen Shot 2022-02-21 at 10 15 09 am" src="https://user-images.githubusercontent.com/53250006/154868657-1928c75c-740e-45ee-a8b9-74a5e7320993.png">

# **Output**
As of the current stage, the output is a dictionary with 16 key-value pairs. The first is the time taken to run. The second is the category it predicts. The third is its confidence, or accuracy measure. The 4th to 16th is the degree of confidence to which the model believes could fit in all 13 categories. Sample output:
  
{"time": 1.6e-05, "prediction": "rebar", "prediction_score": 1.0, "aerial shots_score": 0.0, "concrete_score": 0.0, "scaffolding_score": 0.0, "indoor furnishings_score": 0.0, "rebar_score": 1.0, "road_score": 0.0, "railway_score": 0.0, "fire damage_score": 0.0, "columns and beams_score": 0.0, "workers_score": 0.0, "plumbing_score": 0.0, "electrical detail_scores": 0.0, "machinery_score": 0.0}
