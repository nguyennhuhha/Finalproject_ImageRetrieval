import numpy as np
import cv2
from PIL import Image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, resnet50

from skimage.feature import local_binary_pattern


class MyResnet50:
    def __init__(self):
        super().__init__()
        self.model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.shape = 2048 # the length of the feature vector

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        features = self.model.predict(x)
        return features
    
    def extract_features1(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        features = self.model.predict(x)
        return features
    
class MyVGG16:
    def __init__(self):
        super().__init__()
        self.model = vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.shape = 512 # the length of the feature vector

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)
        features = self.model.predict(x)
        return features
    
    def extract_features1(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)
        features = self.model.predict(x)
        return features
