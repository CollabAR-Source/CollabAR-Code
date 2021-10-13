# CollabAR-Code
[![Website shields.io](https://img.shields.io/badge/python-3.6%2B-green)](http://shields.io/)  [![Website shields.io](https://img.shields.io/badge/numpy-1.16-yellow)](http://shields.io/) [![Website shields.io](https://img.shields.io/badge/opencv--python-4.1-lightgrey)](http://shields.io/) [![Website shields.io](https://img.shields.io/badge/scikit--image-0.16.2-blue)](http://shields.io/) [![Website shields.io](https://img.shields.io/badge/keras-2.3-red)](http://shields.io/) [![Website shields.io](https://img.shields.io/badge/tensorflow-2.0-brightgreen)](http://shields.io/)




This repository contains the introductions and the code of the distortion-tolerant image recognizer and the auxiliary-assisted multi-view ensembler for IPSN 2020 paper ["CollabAR: Edge-assisted Collaborative Image Recognition for Mobile Augmented Reality"]() by [Zida Liu](daliu.github.io), [Guohao Lan](https://guohao.netlify.com/), Jovan Stojkovic, Yunfan Zhang, [Carlee Joe-Wong](https://www.andrew.cmu.edu/user/cjoewong/), and [Maria Gorlatova](https://maria.gorlatova.com/).

Before running the scripts in this repository, you should **download the repository** and install the necessary tools and libraries on your computer, including open-cv, skimage, numpy, keras, tensorflow and sklearn.

If you have any questions on this repository or the related paper, please create an issue or send me an email. Email address: zida.liu AT duke.edu.


**Summary**:

* [Distortion-tolerant-image-recognizer](#1)
* [Auxiliary-assisted multi-view ensembler](#2)
* [Citation](#3)
* [Acknowledgments](#4)


## 1. <span id="1">Distortion-tolerant-image-recognizer</span>
In the CollabAR system, the distortion image recognizer incorporates an image distortion classifier and four recognition experts to resolve the domain adaptation problem caused by the image distortions. As DNNs can adapt to a particular distortion, but not multiple distortions at a time, we need to identify the most significant distortion in the image, and adaptively select a DNN that is dedicated to the detected distortion. The architecture is shown below:

<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/Distortion-tolerant.PNG" width = "600" height = "260" hspace="70" align=center />

### 1.1 Distortion classifier
Different types of distortion have distinct impacts on the frequency domain features of the original images. Gaussian blur can be considered as having a two-dimensional circularly symmetric centralized Gaussian convolution on the original image in the spatial domain. This is equivalent to applying a Gaussian-weighted, circularly shaped low-pass filter on the image, which filters out the high-frequency components in the original image. Motion blur can be considered as a Gaussian-weighted, directional low-pass filter that smooths the original image along the direction of the motion. Lastly, the Fourier transform of additive Gaussian noise is approximately constant over the entire frequency domain, whereas the original image contains mostly low-frequency information. Hence, an image with severe Gaussian noise will have higher signal energy in the high-frequency components. We leverage these distinct frequency domain patterns for distortion classification. 


#### 1.1.1 The pipeline of the image distortion classifier
<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/DistortionClassification.PNG" width = "500" height = "100" hspace="150" align=center />

The figure above shows the pipeline of the image distortion classification. First, we convert the original RGB image into grayscale using the standard Rec. 601 luma coding method. Then, we apply the two-dimensional discrete Fourier transform (DFT) to obtain the Fourier
spectrum of the grayscale image, and shift the zero-frequency component to the center of the spectrum. The centralized spectrum is
used as the input for a shallow CNN architecture.

#### 1.1.2 Image distortion classifier training
The training script is provided via https://github.com/CollabAR-Source/CollabAR-Code/blob/master/trainDisClassifer.py. You only need to provide a pristine image dataset for training the classifier because this script can automatically generate *Motion blur*, *Gaussian blur*, and *Gaussian noise* images. Default distortion levels of generated distorted images are the same as those in the IPSN paper reference above. You can change them in the script for your needs.

To train the distortion classifier, follow the procedure below:

1. Prepare your training set and put it in the *CollabAR-Code* dir. Note that the dataset folder cannot contain non-image files.
2. Change the current directory to the *CollabAR-Code* dir.
3. Run the script as follows: `python .\trainDisClassifer.py -training_set`
   - *training_set*: indicates dir that contains the images for training.
4. The generated weights named "*type_model.hdf5*" will be saved in a created folder named "*weights*".

### 1.2 Recognition experts
Based on the output of the distortion classifier, one of the four dedicated recognition experts is selected for image recognition. Here, we use the lightweight MobileNetV2 as the base DNN model for training the recognition experts.

#### 1.2.1 Recognition experts training
When training the experts, all the CNN layers are first initialized with the values trained on the full ImageNet dataset. Then, we use pristine images in the target dataset to train a **pristine expert**. Finally, we fine-tune the pristine expert to get **Motion blur expert**, **Gaussian blur expert**, and **Gaussian noise expert**. During the fine-tuning, half of the images in the mini-batch are pristine, and the other half are distorted with a random distortion level.

The training script is provided via https://github.com/CollabAR-Source/CollabAR-Code/blob/master/trainExpert.py. Default distortion levels for training the recognition experts are the same as that in the IPSN paper referenced above. You can change them in the script for your needs. To train the recognition experts, follow the procedure below:

1. Change the current directory to the *CollabAR-Code* dir.
2. Prepare the training set, the validation set, and the testing set. 
The file tree for training:
```
└───trainExpert.py
└───train
│   └───class0
│       │   image0.jpg
│       │   image1.jpg
│       │   ...
│   └───class1
|   └───class2
│   │   ...
└───validation
└───test
```
3. Run the script as follows: `python .\trainExpert.py -expert_type`
   - *expert_type*: the type of the expert, i.e., *pristine* for the pristine expert, *MB* for Motion blur expert, *GB* for Gaussian blur expert, *GN* for Gaussian noise expert.

An example of training a Gaussian noise expert:
   - Run the script as follows: `python .\trainExpert.py pristine.`
   - The generated weights named "*pristine_expert.hdf5*" will be saved in a created folder named "*weights*".
   - Run the script as follows: `python .\trainExpert.py GN.`
   - The generated weights named "*Gaussian_noise_expert.hdf5*" will be saved in a created folder named "*weights*".
   
## 2. <span id="2">Auxiliary-assisted multi-view ensembler</span>  
CollabAR aggregates the recognition results of the spatially and temporally correlated images to improve the recognition accuracy of the current image. However, given the heterogeneity of the *m* images (i.e., images are captured from different angles, suffer from different distortions with different distortion levels), the images lead to unequal recognition performance. To quantify their performance and help the ensembler in aggregating the results dynamically, auxiliary features can be used. 


### 2.1 The architecture of the Auxiliary-assisted multi-view ensembler
The architecture of the Auxiliary-assisted multi-view ensembler is shown below:
<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/Ensemble.PNG" width = "500" height = "280" hspace="180" align=center />
The normalized entropy ***S*** measures the recognition quality and the confidence of the distortion tolerant image recognizer on the recognition of an image instance. ***P*** is the probability vector of the base expert on an image instance. We propose to use the normalized entropy  ***S*** as the auxiliary feature to help aggregate the probability vectors ***P***  of the base experts. You can find details about how to calculate ***S*** and how to aggregate ***P*** with the help ***S*** in the IPSN paper referenced above.

### 2.2 The Auxiliary-assisted multi-view ensembler inference
The training script is provided via https://github.com/CollabAR-Source/CollabAR-Code/blob/master/multiUser_inference.py.

1. Change the current directory to the *CollabAR-Code* dir.
2. Run the script as follows: `python .\multiUser_inference.py -multi-view-folder`
   - *multi-view-folder*: indicates dir that contains the multi-view images.
3. You can see the *confidence scores*, *inference result* of each image in the folder, and also the aggregated *inference result* of all the images in the folder.

As an example, we provide a group of multi-view images for the inference, you can find them in the *multi_view_images* folder in this repository.
   - Run the script as follows: `python multiUser_inference.py .\multi_view_images.`
   - You can see the result below:
   <img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/EnsembleResult.PNG" width = "320" height = "220" hspace="200" align=center />

## 3. <span id="3">Citation</span>

Please cite the following paper in your publications if the code helps your research. 

     @inproceedings{Liu20CollabAR,
      title={{CollabAR}: Edge-assisted collaborative image recognition for mobile augmented reality },
      author={Liu, Zida and Lan, Guohao and Stojkovic, Jovan and Yunfan, Zhang and Joe-Wong, Carlee and Gorlatova, Maria},
      booktitle={Proceedings of the 19th ACM/IEEE Conference on Information Processing in Sensor Networks},
      year={2020}
    }
    
## 4. <span id="4">Acknowledgments</span>

The authors of the code are [Zida Liu](https://zidaliu.github.io/), [Guohao Lan](https://guohao.netlify.com/), and [Maria Gorlatova](https://maria.gorlatova.com/). This work was done in the [Intelligent Interactive Internet of Things Lab](https://maria.gorlatova.com/) at [Duke University](https://www.duke.edu/).

Contact Information of the contributor: 

* zida.liu AT duke.edu
* guohao.lan AT duke.edu
* maria.gorlatova AT duke.edu

This work is supported by the Lord Foundation of North Carolina and by NSF awards CSR-1903136 and CNS-1908051.
