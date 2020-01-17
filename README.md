# Modak Detection

This project deals predicting whether the given image contains a Modak or not(Image Classification). Modak is an Indian sweet popular in many parts of India.

![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSFl0r0k0fR3Uqle3VKUS3LI8K91eL_ouTolRkvaQZigLLYfcM2)

For this project I have used Residual Neural Network for the Binary Classification because :-  

* ResNets are easy to optimize.
* ResNets improve feature extraction.
* ResNets overcome the vanishing gradient problem.
* ResNets can easily gain accuracy from greatly increased depth, producing results which are better than previous networks.

In this project, I have used ResNet-50 which contains 50 parameter layers.

## Data Collection

I downloaded 217 distinct images of Modak from google and 217 distint Non-Modak images from Kaggle which contained images of Egg, Dairy product, Dessert, Meat, Rice and Fried Food. I got the Non-Modak dataset from Kaggle. Hence, there was a total of 434 images in the dataset, 217 for each class(Modak and Non-Modak).

## Data Augmentation

Since i am working on keras framework I used the "ImageDataGenerator" for augmentation. The implementation is shown in "Augmentation.py" and the Augmented image sample is present in "Augmentation Sample". 

After Augmentation the dataset size increased from 434 to 2064 i.e.1032 images for each class(Modak and Non-Modak).

## Training

I split the dataset as follow:-

* 70% of the dataset for training
* 20% of the dataset for validation
* 10% of the dataset for testing.

As mentioned above, I used the ResNet-50 Architecture for training the model.

## Result

I used the "metrics accuracy_score()" of "sklearn.metrics" package to determine the accuracy and i was able to achieve an accuracy of 97.297%. 
