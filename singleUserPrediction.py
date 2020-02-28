# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:59:10 2019

@author: zida
"""

import numpy as np
import random
import cv2
import os
import sys
from subfunctions.net_framework import create_base_network, build_model
from subfunctions.predicted_type import predicted_type
from subfunctions.nomalize_entropy import nomalize_entropy



def main(imagePath):
	Noise = ['Clear Image','Motion blur', 'Gaussian blur', 'Gaussian Noise']
	classes = ['bag', 'book', 'bottles', 'cup', 'pen', 'phones']

	#build network skeleton
	type_model = create_base_network()
	model = build_model()
	type_model.load_weights('./weights/type_model.hdf5')

	#read the image 
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(224,224))/255
	noiseType = predicted_type(image, type_model)
	print("The noise type is " + Noise[np.argmax(noiseType)]) #show the noise type
	noiseType = Noise[np.argmax(noiseType)]


	if (noiseType == "Clear Image"):
		model.load_weights('./weights/clear_mobile_transfer.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is : ", str(1-nomalize_entropy(Output)))
		print("The result is " + classes[np.argmax(Output)])

	elif (noiseType == "Motion blur"):
		model.load_weights('./weights/M0~40_mobile_transfer.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is ", str(1-nomalize_entropy(Output)))
		print("The result is " + classes[np.argmax(Output)])

	elif (noiseType == "Gaussian blur"):
		model.load_weights('./weights/GB0~41_mobile_transfer.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is ", str(1-nomalize_entropy(Output)))
		print("The result is " + classes[np.argmax(Output)])

	else:
		model.load_weights('./weights/GN0~40_mobile_transfer.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is ", str(1-nomalize_entropy(Output)))
		print("The result is " + classes[np.argmax(Output)])

if __name__== "__main__":
	main(sys.argv[1])
