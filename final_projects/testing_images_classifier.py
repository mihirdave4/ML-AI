from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
model= models.load_model('image_classfire.model')
class_names=['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Sship','Truck']
img= cv.imread('images/car_imahe.jpg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32)) 
plt.imshow(img,cmap=plt.cm.binary)
prediction= model.predict(np.array([img])/255)
index=np.argmax(prediction)
print(f' prediction is {class_names[index]} ')