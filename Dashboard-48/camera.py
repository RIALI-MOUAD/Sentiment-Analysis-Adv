import cv2
from model import FacialExpressionModel
import pandas as pd
import numpy as np
from PIL import ImageFont

global predper
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("Modele/model.json", "Modele/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self,attr=0):
        self.predper = []
        self.video = cv2.VideoCapture(attr)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        self.predper = [np.array([0,0,0,0,0,0,0])]
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)


        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            self.predper = model.predict_emotions(roi[np.newaxis, :, :, np.newaxis])
        return self.predper
