import numpy as np 
from keras.models import load_model
from keras.preprocessing import image

vgg = load_model('./Weights/MobileNet.h5')
img = image.load_img('./boy_sad.jpeg',target_size=(128,128,3))
img = image.img_to_array(img)

print(img.shape)

y = vgg.predict(img.reshape(1,128,128,3))
print(y)