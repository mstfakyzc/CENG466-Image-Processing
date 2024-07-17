"""
Halit Erkan Görgülü
Mustafa Can Akyazıcı
"""

import numpy as np
import cv2 as cv
import os

INPUT_PATH = "./THE3_Images/"
OUTPUT_PATH = "./Outputs/"

# Load the Haar cascade classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')

def read_image(img_path):
    img = cv.imread(img_path)
    return img
    
def write_image(img, output_path):
    cv.imwrite(output_path, img)

def is_skin_color(color):
    # Set the thresholds for the color channels
    #0<=Y<=255 and 135<=Cr<=180 and 85<=Cb<=135
    y_threshold = [0, 255]
    cr_threshold = [135, 180]
    cb_threshold = [85, 135]

    # Check if the color falls within the thresholds
    if y_threshold[0] <= color[0] <= y_threshold[1] and cr_threshold[0] <= color[1] <= cr_threshold[1] and cb_threshold[0] <= color[2] <= cb_threshold[1]:
        return True
    else:
        return False

def detect_faces(image):
	ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
	
	# Use k-means clustering to cluster the pixels into K clusters
	K = 5
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	_, labels, centers = cv.kmeans(ycrcb.reshape(-1, 3).astype(np.float32), K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Convert the labels back into a 2D image
	labels = labels.reshape(ycrcb.shape[:2])
    
    # Initialize a mask to store the skin color clusters
	mask = np.zeros(ycrcb.shape[:2], dtype=np.uint8)
    
    # Iterate over each cluster and determine if it corresponds to skin color
	for k in range(K):
        # Check if the cluster is a skin color cluster
		if is_skin_color(centers[k]):
            # If it is, add it to the mask
			mask[labels == k] = 255
	
    # Convert the image to grayscale
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	gray = cv.equalizeHist(gray)

    # Apply the mask, if provided
	if mask is not None:
		gray = cv.bitwise_and(gray, mask)

    # Detect faces using the Haar cascade classifier 1.11
	faces = face_cascade.detectMultiScale(gray, 1.115, 5)

	#print(faces)
    
    # Draw rectangles around the detected faces
	for (x, y, w, h) in faces:
		cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return image
		
def color_images(img,sample_img):
	#GrayScale color space of sample image
	gsSample = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY)
	#Arrange color map histogram
	colorMap = np.empty((256), dtype=object) 
	counts = np.zeros((256,), dtype=np.uint32)
	indexes = np.zeros((256,), dtype=np.uint32)
	
	#Final color map
	lut = np.zeros((256, 1, 3), dtype=np.uint8)
	
	#Gray scale histogram
	for x in range(sample_img.shape[0]):
		for y in range(sample_img.shape[1]):
			counts[gsSample[x,y]] = counts[gsSample[x,y]] + 1
			
	#Colormap initialization
	for i in range(counts.shape[0]):
		colorMap[i] = np.zeros((counts[i],3), dtype=np.uint8)
	
	#Histogram with contributers as an RGB value
	for x in range(sample_img.shape[0]):
		for y in range(sample_img.shape[1]):
			index = indexes[gsSample[x,y]]
			colorMap[gsSample[x,y]][index] = sample_img[x,y]
			indexes[gsSample[x,y]] = indexes[gsSample[x,y]] + 1
			
	#Assigning color map with the most common rgb value contributing the each bin of grayscale
	for i in range(lut.shape[0]):
		unique, counts = np.unique(colorMap[i], axis=0, return_counts=True)
		if len(unique):
			rgb = unique[counts.argmax()]
			lut[i,0,0] = rgb[0]
			lut[i,0,1] = rgb[1]
			lut[i,0,2] = rgb[2]
		else:
			lut[i,0,0] = i
			lut[i,0,1] = i
			lut[i,0,2] = i
			
	
	#Applying color mapping
	res = cv.LUT(img, lut)
	
	return res
		
	
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    img1 = read_image(INPUT_PATH + "1_source.png")
    img1f = detect_faces(img1)
    write_image(img1f,OUTPUT_PATH + "1_faces.png")
    
    img2 = read_image(INPUT_PATH + "2_source.png")
    img2f = detect_faces(img2)
    write_image(img2f,OUTPUT_PATH + "2_faces.png")
    
    img3 = read_image(INPUT_PATH + "3_source.png")
    img3f = detect_faces(img3)
    write_image(img3f,OUTPUT_PATH + "3_faces.png")
    
    img1 = read_image(INPUT_PATH + "1.png")
    img1s = read_image(INPUT_PATH + "1_source.png")
    colored1 = color_images(img1,img1s)
    write_image(colored1,OUTPUT_PATH + "1_colored.png")
    
    img2 = read_image(INPUT_PATH + "2.png")
    img2s = read_image(INPUT_PATH + "2_source.png")
    colored2 = color_images(img2,img2s)
    write_image(colored2,OUTPUT_PATH + "2_colored.png")
    
    img3 = read_image(INPUT_PATH + "3.png")
    img3s = read_image(INPUT_PATH + "3_source.png")
    colored3 = color_images(img3,img3s)
    write_image(colored3,OUTPUT_PATH + "3_colored.png")
    
    img4 = read_image(INPUT_PATH + "4.png")
    img4s = read_image(INPUT_PATH + "4_source.png")
    colored4 = color_images(img4,img4s)
    write_image(colored4,OUTPUT_PATH + "4_colored.png")
    
