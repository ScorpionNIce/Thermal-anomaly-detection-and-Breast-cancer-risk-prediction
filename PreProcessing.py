# importing the module 
import cv2 
#import sys
import numpy as np
#from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
#from skimage.util import invert



class PreProcessing():
    # function to display the coordinates of 
    # of the points clicked on the image
    def __init__(self):
        self.coordinates = list()
        self.backup = None
        self.title = 'Select ROI by marking points around ROI, undo point selection by pressing right mouse button and hit Enter key to finalise'
        #self.title = 'LMB - plot ROI boundary point; RMB - Erase last plotted boundary point; Enter - Confirm given points'

    def click_event(self,event, x, y, flags, params ): 
        
        # select points 
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x,y), 3, (255, 255, 255), 2)
            self.coordinates.append((x,y))
            cv2.imshow(self.title, self.img)
            ret = cv2.waitKey()
            if ret == 13:
                cv2.destroyAllWindows()
                return
      
        # removing selected points      
        if event==cv2.EVENT_RBUTTONDOWN:
            if len(self.coordinates)>0:
                x,y = self.coordinates.pop()
                #y,x = self.coordinates.pop()
            
            start_x = x-4
            start_y = y-4

            #print(start_x, start_y)
            
            for i in range(10):
                if start_y+i<0 or start_y+i>479:
                    continue
                for j in range(10):
                    if start_x+j<0 or start_x+j>639:
                        continue
                    self.img[start_y+i][start_x+j] = self.backup[start_y+i][start_x+j]

            cv2.imshow(self.title, self.img)
            ret = cv2.waitKey()
            if ret == 13:
                cv2.destroyAllWindows()
                return
            #print(coordinates_left)
            
      
    # driver function 
    #if __name__=="__main__": 
    def get_ROI(self,imageBGR):
        '''
        This function returns a mask for ROI.
        '''
        # reading the image
        self.img = imageBGR.copy()
        #self.img = cv2.resize(self.img, (640, 480), interpolation = cv2.INTER_LANCZOS4)
        mask = np.zeros((480, 640), np.uint8)  

        #global backup
        self.backup = self.img.copy()
        
        # displaying the image
        ##Select ROI by marking points around ROI and hit Enter key to finalise
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, self.img) 
      
        # setting mouse hadler for the image 
        # and calling the click_event() function 
        cv2.setMouseCallback(self.title, self.click_event)

        # wait for a key to be pressed to exit 
        cv2.waitKey()
      
        # close the window 
        cv2.destroyAllWindows() 
        
        if len(self.coordinates)==0:
            return None
        
        #mask creation
        pts = np.array(self.coordinates, np.int32)
        isClosed = True
        #color = (255, 0, 0)
        color = (255, 255, 255)
        #thickness = 2
        thickness = 1

        #Creating lines connecting points on the image
        pts = pts.reshape((-1, 1, 2))
        image = cv2.polylines(mask, [pts],  isClosed, color, thickness)
        #mask3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #cv2.imwrite("mask2.jpg", mask3d)

        #mask filling
        ellipse_points = convex_hull_image(mask)
            
        result = np.zeros(shape = [480, 640], dtype = np.uint8)
        size = mask.shape
        for i in range(size[0]):
            for j in range(size[1]):
                if ellipse_points[i][j] == True:
                    result[i][j] = 255

        #plt.show()
        #cv2.imshow('image',img)
        #cv2.imshow('mask',mask)
        #cv2.imshow('result', result)
        #cv2.waitKey()
        return result


