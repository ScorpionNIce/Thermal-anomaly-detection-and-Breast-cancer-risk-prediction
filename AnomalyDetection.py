#BGR
import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

class AnomalyDetection():

    def __init__(self, original, image, grayLevel):
        self.orig1 = original.copy()
        self.orig2 = original.copy()
        self.orig = original.copy()
        self.roi = image.copy()
        self.grayLevel = grayLevel
        #print(grayLevel)

    def grayscale(self, image, flag = 0):
        if flag == 1:
            coefficients = [-0.2,0.6,-0.2]
        else:
            coefficients = [0.1,0.6,0.1]
        # Gives blue channel all the weight
        # for standard gray conversion, coefficients = [0.114, 0.587, 0.299]
        m = np.array(coefficients).reshape((1,3))
        gray = cv2.transform(image, m)
        return gray


    def detect(self):
        '''
            This function first applies details enhancment function to
            the roi image which is followed by k means clustering. After
            that we are converting the image into gray scale to apply
            threshold value. The value used is 15%. This is used to find
            the anomaly within the image. Once the anomaly is found then
            we are just highlighting the pixel for cold and hot anomaly i.e red and blue.
        '''
        #image = cv2.resize(self.roi, (640, 480), interpolation = cv2.INTER_LANCZOS4)
        #self.orig = cv2.resize(self.orig, (640, 480), interpolation = cv2.INTER_LANCZOS4)
        image = self.roi
        img_enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        
        # convert to RGB
        img = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        
        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        gray = self.grayscale(res2)

##        coefficients = [0.1,0.6,0.1]
##        # Gives blue channel all the weight
##        # for standard gray conversion, coefficients = [0.114, 0.587, 0.299]
##        m = np.array(coefficients).reshape((1,3))
##        gray = cv2.transform(res2, m)
        avg_low = self.grayLevel[0]
        avg_high = self.grayLevel[1]
        #finding the anomaly pixels 
        position_1 = list()
        position_2 = list()
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if gray[i][j] > 0 :
                    temp = avg_low - gray[i][j]
                    if temp > avg_low*15/100:
                        position_1.append((i,j))

        lowerbound = 1500
        #print(len(position_1))
        if len(position_1)<lowerbound:
            gray = self.grayscale(res2, 1)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if gray[i][j] > 0 :
                        temp = avg_high - gray[i][j]
                        if temp > avg_high*15/100:
                            position_2.append((i,j))
        #cv2.imshow('gray',gray)
        #cv2.waitKey()
        #highligting the pixels
        for x,y in position_1:
            if self.orig1[x][y][0] < self.orig1[x][y][2]:
                #red
                self.orig1[x][y] = [100,100,200]
                self.orig[x][y] = [100,100,200]
            elif self.orig1[x][y][0] > self.orig1[x][y][2]:
                #blue
                self.orig1[x][y] = [200,100,100]
                self.orig[x][y] = [200,100,100]

        #print(len(position_2))
        for x,y in position_2:
            if self.orig2[x][y][0] < self.orig2[x][y][2]:
                #red
                self.orig2[x][y] = [0,0,0]
                self.orig[x][y] = [100,100,200]
            elif self.orig2[x][y][0] > self.orig2[x][y][2]:
                #blue
                self.orig2[x][y] = [250,250,250]
                self.orig[x][y] = [200,100,100]
                
        return self.orig

