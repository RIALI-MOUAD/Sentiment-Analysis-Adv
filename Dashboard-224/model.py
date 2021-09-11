from tensorflow.keras.models import model_from_json
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']#,'Uncer']
    
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)


    def predict_emotion(self, img):
        L=[]
        self.preds = self.loaded_model.predict(img)
        preds = self.preds
        emotions = np.array(FacialExpressionModel.EMOTIONS_LIST)
        for i in range(len(emotions)):
            pred_max = emotions[np.argmax(preds)]
            #dictio[pred_max]=np.max(preds)
            emotions = np.delete(emotions,np.argmax(preds))
            preds = np.delete(preds,np.argmax(preds))
            L.append(pred_max)
        return L#FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
        
                
    def predict_emotions(self, img):
        self.preds = self.loaded_model.predict(img)
        self.pourcentage = self.preds/np.sum(self.preds) * 100
        #self.pourcentage= np.sort(self.pourcentage)
        return self.pourcentage
