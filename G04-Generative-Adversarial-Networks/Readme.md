# Generating Random Human Faces Using Deep Convolutional GAN (DCGAN)

This project focuses on building and training a **Deep Convolutional GAN (DCGAN)** to implement a face generation model using PyTorch.

## Team Members
- Sepideh Bahrami (960122680003)
- Tahereh Hemmati (960122680017)
- Hesam Ghadimi (960122681003)

<!-- #region -->
**Table of Contents**

[TOCM]

[TOC]

## GANs : A Brief Overview
Generative Adversarial Networks (GANs) belong to the set of generative models (Goodfellow, et
al.,2014), which means they can produce/generate new content.   
The GAN model consists of two networks:

- A generative network G(.) that takes random input z and returns x_g=G(z) that should follow the targeted probability distribution.

- A discriminator network D(.) that takes image vector x_image and classifies whether thegenerated image is real or generated.

The generator needs to learn how to create data so that discriminator cannot distinguish it as fake. The discriminator network has the task to determine if the image is real or fake. An intuitive way to understand GAN is to imagine a forger trying to create a faux Picasso painting (Chollet, n.d.). At first, the forger (generator) is pretty bad at this task. As time goes on, the forger becomes increasingly competent at imitating Picasso's style, and the art dealer becomes increasingly expert at spotting fakes. In the end, they have on their hands some excellent fake Picassos. That’s what a GAN is: a forger network and an expert network, each being trained to best the other.


### Face Generation
In this project, we defined and trained a DCGAN on a dataset of faces (CelebA). Our goal is to get a generator network to generate *new* images of faces that look as realistic as possible!
There are a series of tasks in this project as following:

- Loading and preprocessing data.
- Defining and training DCGAN.
- At the end of the project, we'll be able to visualize the result of trained DCGAN.
- The generated samples should look like fairly realistic faces with small amounts of noise.

### Brief Video Description
The video description of this project can be accessed [here.](https://drive.google.com/file/d/1N-Ti4Ld38tkF5R58DFTBPGfJmM4SW61X/view?usp=sharing)




### Dataset
The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since the goal is generating faces, the annotations won't be needed; the images are only required. Note that these are colored images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.

![Screenshot (104)](https://user-images.githubusercontent.com/73435225/104035725-b885be00-51e7-11eb-84d9-757892f71d54.png)
                                          
                                          samples of CelebA dataset
### Access to The dataset
You can access to the CelebA dataset used in this project from the following link:
- [Link of download CelebA dataset ](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)

**Notes about above dataset**:
- The dataset retrieved from the above link contains only 32600 images of CelebA dataset.
- These images are rescaled to 64x64 pixcels. 


<!-- #endregion -->

<!-- #region -->
## Usage
This project has been implemented in`Face Generation Project.ipynb`using Google Colab; therefore, all dependencies are satisfied by running the mentioned notebook on Google Colab.

[**This link**](https://drive.google.com/file/d/15d3v3y3__CSsOJxUmLpy3Odklmqjzk3z/view?usp=sharing) opens the notebook on Google Colab.

### Prepration
Before running the notebook, create the following folders in the Google Colab session storage (content).

	...
	├──gifs
	├──plots
	├──training_samples
	...

## Results


![lr_0 0002_beta1_0 50_bs_32_epochs_20](https://user-images.githubusercontent.com/73435225/104039844-27fdac80-51ec-11eb-9521-8edc05384ff6.gif)

![lr_0 0002_beta1_0 50_bs_32](https://user-images.githubusercontent.com/73435225/104040775-53cd6200-51ed-11eb-9de2-0383b0f089f0.png)


## Acknowledgement

- [**Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**](https://arxiv.org/abs/1511.06434)
- [**NIPS 2016 Tutorial: Generative Adversarial Networks**](https://arxiv.org/abs/1701.00160)

- [**DCGAN--Image Generation**](https://www.researchgate.net/publication/330983916_DCGAN--Image_Generation)

- [**DCGAN TUTORIAL**](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
<!-- #endregion -->

```python

```
