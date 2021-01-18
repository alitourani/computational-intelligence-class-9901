# Deep Residual Network (DRN)

## Team Members:
- Sadaf Oraji (960122681004)
- Mohaddeseh Rafiei (960122681015)
- Fatemeh Moradian (950122680034)

## Project Description:
This model uses the pytorch library on the CIFAR10 data set, which contains 60,000 color images measuring 32 * 32 in 10 classes. There are 50,000 photos in the train set and 10,000 photos in the test set. 
This model is designed using a residual neural network architecture and uses a pre-trained ResNet50 model on the ImageNet dataset for implementation.
In order for the model to be able to practice on our data set, a fully connected layer has been added to increase the output size of the ResNet50 model from 128 to 10, which is the number of classes in our data set.




 

Now we randomly select ten photos from the train set and give them to the model to recognize their category, and out of ten, our model gives the correct answer to nine:
[1] (https://s17.picofile.com/file/8421870926/6B46E416_4113_457B_9DE0_F63503AED5F4.jpeg)


## Result:

 [2] (https://s17.picofile.com/file/8421871868/C78726D7_1C64_4B14_AC39_E5B461BEA65A.jpeg)
