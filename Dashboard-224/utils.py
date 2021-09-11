from flask import Flask, render_template, request, redirect,url_for
from werkzeug.utils import secure_filename
from camera import VideoCamera
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px

UPLOAD_FOLDER = 'datasets'
VIDEO_FOLDER = 'Videos'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'png', 'jpg', 'jpeg', 'gif','mp4'}
VIDEO_EXTENSIONS = {'mkv','mp4'}
emotions = ['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_vid(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def gen(camera,filename,length):
    df = pd.DataFrame(columns=emotions)
    i = 0
    print("---------------------------")
    while True:
        L = camera.get_frame()
        print(L[0])
        df.loc[i] = L[0]
        #print(L[0])
        df.to_csv(UPLOAD_FOLDER+"/"+filename+".csv")
        i=i+1
        print(100*i//length,'%-------------------')
        render_template("download.html", value=100*i//length)


def Nan(i):
    if i == None:
        return 'Nan'
    return i

def df_to_arrays(df):
    L = []
    for col in df.columns.values:
        L.append(list(df[col].values))
    return L

def pie_chart(df):
    L = []
    for i in range(df.shape[0]):
        if np.max(df.loc[i].values) != 0:
            L.append(df.columns[np.argmax(df.loc[i].values)])
        else :
            L.append(None)
    data = pd.DataFrame(data={
        'values':list(np.ones(df.shape[0])),
        'names':L
    })
    dataArr = df_to_arrays(data)
    return dataArr

def radar_chart(df):
    my_dict = {Nan(i): np.mean(df[i].values) for i in df.columns.values[:-1]}
    r = [my_dict[i] for i in df.columns.values[:-1]]
    theta=list(my_dict.keys())
    return [r,theta]
def mean_data(df):
    arr ,L= df_to_arrays(df),[]
    for i in range(len(arr)):
        L.append(round(np.mean(arr[i]), 2))
        #print("%.2f" % np.mean(arr[i]))
    L = list(np.array(L)/np.sum(np.array(L)) * 100)
    return [round(i,2) for i in L]
