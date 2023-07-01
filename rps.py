# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model("keras_model.h5")



vid = cv2.VideoCapture(0)
  
while(True):
      
    ret, frame = vid.read()

    image=cv2.resize(frame,(224,224))

    testImage=np.array(image,dtype=np.float32)
    testImage=np.expand_dims(testImage,axis=0)
    
    normalizeImage=testImage/255.0
    

    prediction=model.predict(normalizeImage)
    
    print(prediction)

    cv2.imshow('frame', frame)
      
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
vid.release()

cv2.destroyAllWindows()