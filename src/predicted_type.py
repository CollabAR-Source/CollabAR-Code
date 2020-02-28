import cv2
import numpy as np

def _preprocess(image):
    return (image-image.min())/(image.max()-image.min())

def _get_Fourier(image):
    f = np.fft.fft2(image)
    fshif = np.fft.fftshift(f)
    s1 = np.log(20+np.abs(fshif))
    s1 = _preprocess(s1)
    return s1 

def _rgb2gray(rgb):
	image = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
	image = _preprocess(image)
	return image

def predicted_type(image, trained_model):
    temp_image = _get_Fourier(_rgb2gray(image))[:,:,np.newaxis]
    temp = trained_model.predict(temp_image[np.newaxis,:,:,:])
    return temp