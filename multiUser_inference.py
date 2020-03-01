import numpy as np
import random
import cv2
import os
import sys
from src.net_framework import create_base_network, build_model
from src.predicted_type import predicted_type
from src.nomalize_entropy import nomalize_entropy

Noise = ['Clear Image','Motion blur', 'Gaussian blur', 'Gaussian Noise']
classes = ['bag', 'book', 'bottles', 'cup', 'pen', 'phones']

def singleUser(imagePath, type_model, model):

	#read the image 
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(224,224))/255
	noiseType = predicted_type(image, type_model)
	print("-------------------")
	print("The noise type is " + Noise[np.argmax(noiseType)]) #show the noise type
	noiseType = Noise[np.argmax(noiseType)]


	if (noiseType == "Clear Image"):
		model.load_weights('./weights/pristine_expert.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is : ", str(1-nomalize_entropy(Output)))
		print("This image is a " + classes[np.argmax(Output)])

	elif (noiseType == "Motion blur"):
		model.load_weights('./weights/motion_blur_expert.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is ", str(1-nomalize_entropy(Output)))
		print("This image is a " + classes[np.argmax(Output)])

	elif (noiseType == "Gaussian blur"):
		model.load_weights('./weights/Gaussian_blur_expert.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is ", str(1-nomalize_entropy(Output)))
		print("This image is a " + classes[np.argmax(Output)])

	else:
		model.load_weights('./weights/Gaussian_noise_expert.hdf5')
		if image.mean()>1:
			image = image/255
		Output = model.predict(image[np.newaxis]) #get original prob vector
		print("The confidence score is ", str(1-nomalize_entropy(Output)))
		print("This image is a " + classes[np.argmax(Output)])

	print("-------------------")
	return Output*(1-nomalize_entropy(Output))

def main(files):

	#build network skeleton
	type_model = create_base_network()
	model = build_model()
	type_model.load_weights('./weights/type_model.hdf5')
	#esemble framework
	esemble_result = 0
	file = os.listdir(files)
	for iterm in file:
		print("This is image " + iterm)
		esemble_result = esemble_result + singleUser(files+iterm, type_model, model)

	print("The ensemble result of all the images is a " + classes[np.argmax(esemble_result)])

if __name__== "__main__":
	main(sys.argv[1])
