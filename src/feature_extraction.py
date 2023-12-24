import numpy as np

from keras.utils import img_to_array
from keras.preprocessing import image
from keras.applications import vgg16, resnet50, xception, efficientnet_v2
from keras.models import Model

class MyResnet50:
    def __init__(self):
        super().__init__()
        self.model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.shape = 2048 # the length of the feature vector

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        features = self.model.predict(x)
        return features
    
    def extract_features1(self, img):
        img = img.resize((224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x) # Predict with shape (1, 2048)
        features = self.model.predict(x)
        return features

class MyVGG16:
    def __init__(self):
        super().__init__()
        self.model = vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.shape = 512 # the length of the feature vector

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)
        features = self.model.predict(x) # Predict with shape (1, 512)
        return features
    
    def extract_features1(self, img):
        img = img.resize((224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)
        features = self.model.predict(x)
        return features
    
class MyXception:
    def __init__(self):
        super().__init__()
        self.model = xception.Xception()
        self.shape = 1000 # the length of the feature vector

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception.preprocess_input(x)

        feature = self.model.predict(x)  # Predict with shape (1, 1000) 
        feature = feature / np.linalg.norm(feature)  # Normalize

        return feature
    
    def extract_features1(self, img):
        img = img.resize((299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception.preprocess_input(x)

        feature = self.model.predict(x)
        feature = feature / np.linalg.norm(feature)  # Normalize
        
        return feature
    
class MyEfficient:
    def __init__(self):
        super().__init__()
        base =  efficientnet_v2.EfficientNetV2L()
        self.model = Model(inputs=base.input, outputs=base.get_layer('top_dropout').output)
        self.shape = 1280 # the length of the feature vector

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(480, 480))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = efficientnet_v2.preprocess_input(x)

        feature = self.model.predict(x)  # Predict with shape (1, 1280) 
        feature = feature / np.linalg.norm(feature)  # Normalize

        return feature
    
    def extract_features1(self, img):
        img = img.resize((480, 480))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = efficientnet_v2.preprocess_input(x)

        feature = self.model.predict(x)
        feature = feature / np.linalg.norm(feature)  # Normalize
        
        return feature