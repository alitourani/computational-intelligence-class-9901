# Deep Convolutional Network (DCN)

## Team Members:
- Seyyedeh Zakieh Es'haghi (960122680033)
- Kosar Gholam Alizadeh (960122680041)

## Project Description:
In this project,we are going to classify human pictures according to their gender.We implemented a Deep Convolutional Network.

## Dataset
We used celebA datsetset which includes about 200K images of celebrities.We used 50K pictures because it is enough and we could train the network such took at least accuracy of 90% on test set.
We splited the data into three pieces:Train set,validation set and test set.


## Training phase
We trained the model for 50 epochs and batch size 64.Also we saved the model in the maximum accuracy of validation set on hard disk.We are going to use this weights as the best weights when testing the model in evaluation phase.

## Evaluating phase
We tested the model on test set and could achieve accuracy of 93.6%.
