
"""
Created on Tue Sep  1 12:27:58 2020

@author: Patryk
"""


import numpy as np
import cv2
import pandas as pd
from sklearn.feature_selection import SelectKBest,chi2
import pickle
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import imageio
import matplotlib.pyplot as plt
from skimage import io, color, restoration, img_as_float
from skimage.feature import hog
from skimage import data, exposure
from PIL import Image
from skimage.color import rgb2gray
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import os
 
def features(Im):
    data = pd.DataFrame()

    Imm = Im.reshape(-1)
    data['Original Image'] = Imm


    counter = 1
    kernel = []
    for theta in range(3):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0,np.pi, np.pi / 4):
                for gamma in (0.02, 0.6):
               
                
                    gabor = 'F.Gabor' + str(counter)
                    
                    rozmiar=9
                    kernels= cv2.getGaborKernel((rozmiar, rozmiar), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernel.append(kernels)
                    
                    filterG = cv2.filter2D(Imm, cv2.CV_8UC3, kernels)
                    ImageWithFilter = filterG.reshape(-1)
                    data[gabor] = ImageWithFilter 
                    counter += 1
                    




    return data
#loads the trained model
load= pickle.load(open("PCA_Hybrid_150estimators_depth20_split2_PCA_20cech", 'rb'))
folder1 = "C:/Users/Patryk/Desktop/lysoci/"
path_Mask="C:/Users/Patryk/Desktop/MaskiSkalowane/"
folder = os.listdir(folder1)

for plik in  folder:  #Biorę każdy obraz z folderu do segmentacji
    print(plik)
    img1= cv2.imread(folder1+plik)
    img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) # Convert images to gray scale
    pixels1 = img.reshape(-1)
    X = features(img)
    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
    X=np.column_stack  ((pixels1,X))
    result = load.predict(X)
    posegmentowane = result.reshape((img.shape))
    #saves the segmented images at the specified location
    io.imsave("/home/pczernik/RF_RFE_30cech/"+ obrazyy, (posegmentowane*255).astype(np.uint8))   
