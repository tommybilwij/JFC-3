**File structure**
Each directory contains a minimum of 4 files (Eedorenko, 2020): 
- Build.sh
- DockerFile
- Requirements.txt
- A Python file (.py)

Build.sh and Dockerfile contain client code. Build.sh acquires the directory and creates a Docker Image (DI) for the component. This is the containerised file that is pushed onto Kubeflow. Meanwhile, the "Dockerfile" file includes the command that the container needs to execute to perform its function. 

Requirements.txt contains the custom Python packages required to run the component. This will be downloaded via PIP.

The Python file contains the functionalities of each component, thus making up the majority of the runtime code for each component. Each python file is what is described above in section 3.1.

<!-- Special files in Deploy component (component 3.1.4)
 -->


**Using the Pipeline**
pipeline.py must be executed in order to create pipeline.py.tar.gz, a file required by Kubernetes. This scoring URL can be found once we publish the pipeline to AKS using the Kubeflow dashboard and successfully execute the experiment.

http://33830fd9-e1ed-44e0-958f-ba6be7c798be.eastus.azurecontainer.io/score?image=<Image URL>

Then replace “<Image URL>” with the image you wish to classify.

**Output**
As of the current stage, the output is a dictionary with 16 key-value pairs. The first is the time taken to run. The second is the category it predicts. The third is its confidence, or accuracy measure. The 4th to 16th is the degree of confidence to which the model believes could fit in all 13 categories. Sample output:
  
{"time": 1.6e-05, "prediction": "rebar", "prediction_score": 1.0, "aerial shots_score": 0.0, "concrete_score": 0.0, "scaffolding_score": 0.0, "indoor furnishings_score": 0.0, "rebar_score": 1.0, "road_score": 0.0, "railway_score": 0.0, "fire damage_score": 0.0, "columns and beams_score": 0.0, "workers_score": 0.0, "plumbing_score": 0.0, "electrical detail_scores": 0.0, "machinery_score": 0.0}
