import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img

modelPath = 'low_light_image_denoising_model.h5'
testFolder = './test/low'
outputFolder = './test/predicted'

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

model = load_model(modelPath)

def preprocessImage(imgPath, targetSize=(256, 256)):
    img = image.load_img(imgPath, target_size=targetSize)
    imgArray = img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray /= 255.0
    return imgArray

def savePredictionImage(prediction, outputPath):
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.clip(prediction * 255.0, 0, 255).astype('uint8')
    img = array_to_img(prediction)
    img.save(outputPath)

def predictImage(model, imgPath):
    img = preprocessImage(imgPath)
    prediction = model.predict(img)
    return prediction

for imgName in os.listdir(testFolder):
    imgPath = os.path.join(testFolder, imgName)
    if os.path.isfile(imgPath) and imgName.lower().endswith(('png', 'jpg', 'jpeg')):
        prediction = predictImage(model, imgPath)
        print(f'Prediction for {imgName}: {prediction.shape}')
        outputPath = os.path.join(outputFolder, imgName)
        savePredictionImage(prediction, outputPath)
