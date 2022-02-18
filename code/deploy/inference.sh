#!/bin/bash
while getopts "n:w:g:" option;
    do
    case "$option" in
        n ) DEPLOYMENT_NAME=${OPTARG};;
        w ) WORKSPACE=${OPTARG};;
        g ) RESOURCE_GROUP=${OPTARG};;
    esac
done
echo "test the deployment with a concrete image"
az ml service run -n ${DEPLOYMENT_NAME} -d '{ "image": "https://www.techexplorist.com/wp-content/uploads/2017/07/concrete.jpg" }' -w ${WORKSPACE} -g ${RESOURCE_GROUP}
echo "test the deployment with a railway image"
az ml service run -n ${DEPLOYMENT_NAME} -d '{ "image": "http://www.railway-fasteners.com/uploads/allimg/how-to-build-a-railway-track.jpg" }' -w ${WORKSPACE} -g ${RESOURCE_GROUP}
