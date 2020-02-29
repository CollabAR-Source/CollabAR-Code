# CollabAR-Code

This repository contains the introductions and the code of the distortion-tolerant image recognizer and the auxiliary-assisted multi-view ensembler for IPSN 2020 paper ["CollabAR: Edge-assisted Collaborative Image Recognition for Mobile Augmented Reality"]() by [Zida Liu](daliu.github.io), [Guohao Lan](https://guohao.netlify.com/), Jovan Stojkovic, Yunfan Zhang, [Carlee Joe-Wong](https://www.andrew.cmu.edu/user/cjoewong/), and [Maria Gorlatova](https://maria.gorlatova.com/).

Before running the scripts in this repository, you should install the necessary tools and libraries on your computer, including open-cv, skimage, numpy, keras, tensorflow and sklearn.

If you have any questions on this repository or the related paper, please create an issue or send me an email. Email address: zida.liu AT duke.edu.


**Summary**:

* [Distortion-tolerant-image-recognizer](#1)
* [Auxiliary-assisted multi-view ensembler](#2)
* [Citation](#3)
* [Acknowledgments](#4)


## 1. <span id="1">Distortion-tolerant-image-recognizer</span>
In the CollabAR system, the distortion image recognizer incorporates an image distortion classifier and four recognition experts to resolve the domain adaptation problem caused by the image distortions. As DNNs can adapt to a particular distortion, but not multiple distortions at the same time, we need to identify the most significant distortion in the image, and adaptively select a DNN that is dedicated to the detected distortion. The architecture is shown below:

<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/Distortion-tolerant.PNG" width = "600" height = "260" hspace="70" align=center />

### 1.1 Distortion classifier
Different types of distortion have distinct impacts on the frequency domain features of the original images. Gaussian blur can be considered as having a two-dimensional circularly symmetric centralized Gaussian convolution on the original image in the spatial domain. This is equivalent to applying a Gaussian-weighted, circularly shaped low pass filter on the image, which filters out the high-frequency components in the original image. Motion blur can be considered as a Gaussian-weighted, directional low pass filter that smooths
the original image along the direction of the motion. Lastly, the Fourier transform of additive Gaussian noise is approximately
constant over the entire frequency domain, whereas the original image contains mostly low-frequency information. Hence, an
image with severe Gaussian noise will have higher signal energy in the high-frequency components. We leverage these distinct frequency
domain patterns for distortion classification. 


#### 1.1.1 The pipeline of the image distortion classifier
<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/DistortionClassification.PNG" width = "500" height = "100" hspace="150" align=center />

The figure above shows the pipeline of the image distortion classification. First, we convert the original RGB image into grayscale using the standard Rec. 601 luma coding method. Then, we apply the two-dimensional discrete Fourier transform (DFT) to obtain the Fourier
spectrum of the grayscale image, and shift the zero-frequency component to the center of the spectrum. The centralized spectrum is
used as the input for a shallow CNN architecture.

#### 1.1.2 Image distortion classifier training
The training script is provided via https://github.com/CollabAR-Source/CollabAR-Code/blob/master/trainDisClassifer.py. You only need to provide a pristine image dataset for training the classifier because this script can automatically generate *Motion blur*, *Gaussian blur*, and *Gaussian noise* images. Default distortion levels of generated distorted are the same as that in IPSN paper. You can change them in the script for your needs.

To train the distortion classifier, follow the procedure below:

1. Download the training script. Then put the script in the same dir with the training set folder. Note that the training set folder cannot contain non-image files.
2. Run the script as follows: python .\trainDisClassifer.py -training_set
   - *training_set*: indicates dir that contains the training images.
3. The generated weights named "*type_model.hdf5*" will be saved in a created folder named "*weights*".

### 1.2 Recognition experts
Based on the output of the distortion classifier, one of the four dedicated recognition experts is selected for image recognition. Here, we use the lightweight MobileNetV2 as the base DNN model for training the recognition experts.

#### 1.2.1 Recognition experts training
When training the experts, all the CNN layers are first initialized with the values trained on the full ImageNet dataset. Then, we use pristine images in the target dataset to train a pristine expert. Finally, we fine-tune the pristine expert to get motion blur expert, Gaussian blur expert, and Gaussian noise expert. During the fine-tuning, half of the images in the mini-batch are pristine, and the other half are distorted with a random distortion level.

The training script is provided via https://github.com/CollabAR-Source/CollabAR-Code/blob/master/trainExpert.py. Default distortion levels for training the recognition experts are the same as that in IPSN paper. You can change them in the script for your needs. To train the recognition experts, follow the procedure below:

1. Download the training script and put it in the same dir with the training set, the validation set and the testing set. 
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
2. Run the script as follows: python .\trainExpert.py -expert_type
   - *expert_type*: the type of the expert, i.e., *pristine* for the pristine expert, *MB* for motion blur expert, *GB* for Gaussian blur expert, *GN* for Gaussian noise expert.

An example of training Gaussian noise expert:
   - Run the script as follows: python .\trainExpert.py pristine.
   - The generated weights named "*pristine_expert.hdf5*" will be saved in a created folder named "*weights*".
   - Uncomment the 113th line in the script.
   - Run the script as follows: python .\trainExpert.py GN.
   - The generated weights named "*Gaussian_noise_expert.hdf5*" will be saved in a created folder named "*weights*".
   
