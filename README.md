# CollabAR-Code

This repository contains the introductions and the code of the distortion-tolerant image recognizer and the auxiliary-assisted multi-view ensembler for IPSN 2020 paper ["CollabAR: Edge-assisted Collaborative Image Recognition for Mobile Augmented Reality"]() by [Zida Liu](daliu.github.io), [Guohao Lan](https://guohao.netlify.com/), Jovan Stojkovic, Yunfan Zhang, [Carlee Joe-Wong](https://www.andrew.cmu.edu/user/cjoewong/), and [Maria Gorlatova](https://maria.gorlatova.com/).

**Summary**:

* [Distortion-tolerant-image-recognizer](#1)
* [Auxiliary-assisted multi-view ensembler](#2)
* [Citation](#3)
* [Acknowledgements](#4)


## 1. <span id="1">Distortion-tolerant-image-recognizer</span>
In the CollabAR system, the distortion image recognizer incorporates an image distortion classifier and four recognition experts to resolve the domain adaptation problem caused by the image distortions. As DNNs can adapt to a particular distortion, but not multiple distortions at the same time, we need to identify the most significant distortion in the image, and adaptively select a DNN that is dedicated to the detected distortion. The architecture is shown below:

<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/Distortion-tolerant.PNG" width = "600" height = "260" hspace="70" align=center />

### 1.1 Distortion classifier
Different types of distortion have distinct impacts on the frequency domain features of the original images. Gaussian blur can be considered as having a two-dimensional circularly symmetric centralized Gaussian convolution on the original image in the spatial domain. This is equivalent to applying a Gaussian-weighted, circularly shaped low pass filter on the image, which filters out the high frequency components in the original image. Motion blur can be considered as a Gaussian-weighted, directional low pass filter that smooths
the original image along the direction of the motion. Lastly, the Fourier transform of additive Gaussian noise is approximately
constant over the entire frequency domain, whereas the original image contains mostly low frequency information. Hence, an
image with severe Gaussian noise will have higher signal energy in the high frequency components. We leverage these distinct frequency
domain patterns for distortion classification. 


#### 1.1.1 The pipeline of the image distortion classifier
<img src="https://github.com/CollabAR-Source/CollabAR-Code/blob/master/figures/DistortionClassification.PNG" width = "500" height = "100" hspace="150" align=center />

The figure above shows the pipeline of the image distortion classification. First, we convert the original RGB image into grayscale using the standard Rec. 601 luma coding method. Then, we apply the twodimensional discrete Fourier transform (DFT) to obtain the Fourier
spectrum of the grayscale image, and shift the zero-frequency component to the center of the spectrum. The centralized spectrum is
used as the input for a shallow CNN architecture.

#### 1.1.2 Image distortion classifier training
To train the distortion classifier, follow the procedure below:

1. Before running the script, you should install the necessary tools and libraries on your computer, including: open-cv, skimage, numpy, keras, tensorflow and sklearn. The versions of these libraries are shown at the very beginning of *README*. 
2. Then, put the script under the folder ''MVMDD''.
3. Run the script as follows: `python .\distortion_generation.py -source_dir -distortion_type -distortion_degree`
   - *source_dir*: indicates the original dir that contains the pristine images.
   - *distortion_type*: indicates the type of distortion you would like to synthesize. There are three options available: 
      - *MB* for motion blur 
      - *GB* for Gaussian blur 
      - *GN* for Gaussian noise
   - *distortion_degree*: indicates the distortion level you would like to set.
 4. The generated images will be saved in the generated folder.


