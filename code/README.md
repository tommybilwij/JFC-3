Using the Pipeline
pipeline.py must be executed in order to create pipeline.py.tar.gz, a file required by Kubernetes. This scoring URL can be found once we publish the pipeline to AKS using the Kubeflow dashboard and successfully execute the experiment.

http://33830fd9-e1ed-44e0-958f-ba6be7c798be.eastus.azurecontainer.io/score?image=<Image URL>

Then replace “<Image URL>” with the image you wish to classify.
