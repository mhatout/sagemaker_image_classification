# Image Classification using AWS SageMaker

Used AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.

- The following tasks are performed:
  1. Used a pretrained Resnet50 model from pytorch vision library
  2. Finetune the model with hyperparameter tuning
  3. Implement Profiling and Debugging with hooks
  4. Deploy the model and perform inference

## Project Set Up and Installation

Enter AWS through the gateway in the course and open SageMaker Studio. Download the starter files.

## Dataset

I used the dog breeds dataset contains images from 133 dog breeds divided into training, testing and validation datasets. Can be downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

### Files Used 

- train_and_deploy.ipynb - This jupyter notebook contains all the code and the steps performed in this project and their outputs.
- hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameters Combination.
- train_model.py - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning.
- endpoint_inference.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations), serialization, deserialization and inferences.
    

## Hyperparameter Tuning

- ResNet50 pretrained model to finetuned.
- A pair of fully connected neural networks layers added to the pretrained model to perform the classification.
- AdamW as an optimizer.
- Hyperparamets tuned:
  * Learning rate- 0.01 to 0.1
  * Batch size - [ 64, 128 ]
  * Weight decay - 0.1x to 10x

Hyperparameter tuning job

![Hyperparameter tuning job](snapshots/Screen%20Shot%202022-08-26%20at%202.49.07%20PM.png)

Training job

![Training job](snapshots/Screen%20Shot%202022-08-27%20at%202.09.27%20AM.png)

## Debugging and Profiling

The plot of the debugging output.

![debugging output](snapshots/Screen%20Shot%202022-08-26%20at%206.36.47%20PM.png)

### Profiler Output

[The profiler report](profiler-report.html).

![Profiler Output](snapshots/Screen%20Shot%202022-08-27%20at%201.51.44%20AM.png)


## Model Deployment
- Model was deployed to a "ml.t2.medium" instance. 
- endpoint_inference.py script is used to setup and deploy our working endpoint.
- For testing purposes ,test image is fed to the endpoint for inference.
- The inference is performed using the Predictor Object. 

![End Point Deployment](snapshots/Screen%20Shot%202022-08-26%20at%206.24.38%20PM.png)
