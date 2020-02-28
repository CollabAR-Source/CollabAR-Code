This repository contains the introductions and the code of the distortion-tolerant image recognizer and the auxiliary-assisted multi-view ensembler. IPSN 2020 paper ["CollabAR: Edge-assisted Collaborative Image Recognition for Mobile Augmented Reality"]() by [Zida Liu](daliu.github.io), [Guohao Lan](https://guohao.netlify.com/), Jovan Stojkovic, Yunfan Zhang, [Carlee Joe-Wong](https://www.andrew.cmu.edu/user/cjoewong/), and [Maria Gorlatova](https://maria.gorlatova.com/).


# Distortion-tolerant-image-recognizer
In the CollabAR system, the distortion image recognizer incorporates an image distortion classifier and four recognition experts to resolve the domain adaptation problem caused by the image distortions. As DNNs can adapt to a particular distortion, but not multiple distortions
at the same time, we need to identify the most significant distortion in the image, and adaptively select a DNN that is dedicated to the detected distortion. 

