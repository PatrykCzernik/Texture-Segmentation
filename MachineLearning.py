
"""
Created on Tue Sep  1 12:27:05 2020

@author: Patryk
"""
import numpy as np
import cv2
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import os
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import seaborn as sns


DataFromImages = pd.DataFrame()   # Create data frame
inputI = "C:/Users/Patryk/Desktop/imagesSkalowane/" # Read images
inputIM = "C:/Users/Patryk/Desktop/maskSkalowane/" # Read masks
image =  os.listdir(inputI)  
imageI = image[0:40] 
imageII = image[40:80]
imageIII = image[80:120]
imageIV = image[120:160]
imagesV = image[160:200]
images = [imageI,imageII,imageIII,imageIV,imagesV]
mask =  os.listdir(inputIM)  
mask = mask[0:40] 
maskII = mask[40:80]
maskIII = mask[80:120]
maskIV = mask[120:160]
maskV = mask[160:200]
masks = [maskI,maskII,maskIII,maskIV,maskV]

pca = PCA(n_components=18)
for imagey,maski in zip(images,masks): 
    data = pd.DataFrame() 
    for plik1,mask2 in zip(imagey,maski):
        #print(mask2)
        data = pd.DataFrame()  # It is a momentary data frame that stores the features from each iteration
        data2 = pd.DataFrame() 
        obinputI = cv2.imread(inputI + plik1)  # I read the pictures one by one
        maskinputI = cv2.imread(inputIM + mask2)# I read the masks one by one
        img1 = cv2.cvtColor(obinputI,cv2.COLOR_BGR2GRAY)/255
        pixele1 = img1.reshape(-1)
        label = cv2.cvtColor(maskinputI,cv2.COLOR_BGR2GRAY)/255 
        Piksele = np.array(pixele1)
        etykietki = np.array(label.reshape(-1))
        
        ### Gabor filters Bank
# I generate gabor features in for loops
        counter = 1   #  Use to count gabor features 
        kernel2 = []  # Empty list to keep kernels 
        for theta in range(3):   # Theta: 0, 1/4 . pi and 2/4.pi
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # Sigma: 1 and 3
                for lamda in np.arange(0,np.pi, np.pi / 4):   
                    for gamma in (0.02, 0.6):   #Gamma values: 0.02 and 0.6
                        gabor = 'F.Gabor' + str(counter)  # Name of columns
                        rozmiar=9# Size kernel 9x9
                        kernel = cv2.getGaborKernel((rozmiar, rozmiar), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                        kernel2.append(kernel)
                    #Now filter the image and add values to a new column 
                        filterG = cv2.filter2D(img1, cv2.CV_8UC3, kernel)
                   
                        ImageWithGabor  = filterG.reshape(-1)# Convert 2D image into 1D
                        data[gabor] = ImageWithGabor 
                        counter += 1  # increase by one after each iteration
        # In next step i add values to my  data frame
        filterGgabora = data.values #!!!!!!!!!! It will be  in the future use as X
        InfozMasek = pd.DataFrame(etykietki)  #data frame containing information about masks
        etykiety = InfozMasek.values # !!!!!!!!!!!!!! It will be  in the future use as Y
    
        X=np.column_stack  ((Piksele, filterGgabora)) # link pixels into one whole
        Y = etykiety # The label value is our predicted value
        #pca = PCA(n_components=18)
        Y[Y >= 0.5] = 1
        Y[Y<0.5] = 0
        # Data standardization is a very important step in data processing
        sc = StandardScaler()
        sc.fit(X) 
        X = sc.transform(X)
        X = pca.fit_transform(X) #reduction of data dimensionality by PCA
        
        
        #I split the data into testing and training
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
   
        from sklearn.ensemble import RandomForestClassifier # Import import RandomForestClassifier
        #param_grid = {"n_estimators":[10,50,100,150,200,300,400,500,1000],
        #"max_depth": [5,10,15,20,25],
        #"min_samples_split": [2,5,10,15]}
        #model = RandomForestClassifier()
        #grid_model = GridSearchCV(model,param_grid = param_grid,cv=2)
        #grid_model.fit(X_train, y_train.ravel())
        #param_df = pd.DataFrame.from_records(grid_model.cv_results_['params'])
        #param_df['mean_test_score'] = grid_model.cv_results_['mean_test_score']
        #param_df.sort_values(by=['mean_test_score']).tail()
        model = RandomForestClassifier(warm_start=True,n_estimators = 5,random_state=0,max_depth = 10) # I create my classifier
        model.fit(X_train, y_train.ravel()) # Training model on the trening data
        from sklearn import metrics
        prediction_test = model.predict(X_test)# Testing created model
    
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
Name = "RF_5PCAESplit2"
pickle.dump(model, open(Name, 'wb'))

pca = PCA()
PC = pca.fit_transform(X) 
        

pca = PCA()
PC = pca.fit_transform(X) 
variance = np.cumsum(pca.explained_variance_ratio_)
counter = 0
LK =[]
for i in range(0,49):
    counter = counter+1
    if variance[i]>=0.90:
        LK.append(variance)
print("Number of components",len(LK)) 
    
    

