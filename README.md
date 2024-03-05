[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
# DogBreedClassification

## Project Overview

 Convolutional Neural Networks (CNN) Udacity Data Scientist project. This Project has 2 parts the first is a dog breed classification and the second is a flask web application.
 In the dog breed classification part, a few models were built and tested to try and classify dogs from images to one of 133 dog breeds.
 The Web Application allows the user to upload an image and it checks if it's a dog and returns it's breed, if it's a human and returns the breed the human resembles.

![Sample Output][image1]

## Required Libraries
Python version 3.* is needed.

And the following libraries need to be installed

(1) flask

(2) tensorflow

(3) opencv-python

(4) matplotlib

(5) numpy

(6) glob

If any of them are unavailable, use pip install (library_name)

## Motivation 
I was really interested in how Convolutional Neural Networks work and wanted to learn more. Furthermore, I am really bad at identifying dog breeds, I only know about 5 of them. So, i thought this project would be a great apportionity to
build a model that can identify dog breeds better than me.

## File Describtion
This Repository contains several files and folders.

The run.py file is a web app program that predicts a dog breed from an image supplied by the user. static and templates folders are needed to run this file.

The dog_app.ipynb is a notebook that was used to explore the data and train some models. It saves the best model for each approach in the saved_models folder.

The extract_bottleneck_features.py contains some helper functions to extract bottleneck features.

## Running the Web Application
(1) install all necessary packages.

(2) clone this repository.

(3) Navigate to the repository from the command line.

(4) Run -> python run.py

(5) On your browser, go to http://127.0.0.1:5000

## Analysis 
In this project, I built a InceptionV3 model from a pre-trained model. I was able to achieve an accuracy of 83.01% on my testing dataset when using epochs of 20. In the data sets used There are:

(1) 133 total dog categories.

(2) 8351 total dog images.

(3) 6680 training dog images.

(4) 835 validation dog images.

(5) 836 test dog images.

(6) 13233 human images.

## Results
I was able to get an accuracy of 83.01% on the testing data with the InceptionV3 pre-trained model. And when testing with human images, it was able to detect that it's a human image and give the most resembling dog breed.
It was also successful in producing an error if the provided image is neither a dog's nor a human's.

# Conclusion
Despite having a high accuracy on the testing data set, the results are worse than expected.

to improve the algorithm, I propose the following improvements:

(1) Increase the dataset size. Currently, we only have around 8,000 images across the training, validation, and testing datasets. This number should increase to a minimum of 50,000 images to be able to capture the variety between different breeds and even the variety inside a breed.

(2) trying a different pre-trained model (Resnet50, Xception, or others)

(3) Hyperparameter tuning

(4) increase the number of training epochs

## Licensing, Authors, Acknowledgements
Udacity Data sets: [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 
[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). 
[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz).

