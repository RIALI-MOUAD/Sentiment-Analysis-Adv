from flask import Flask, render_template, Response,request, redirect,url_for
from werkzeug.utils import secure_filename
from camera import VideoCamera
import os,cv2
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
from utils import *

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER



@app.route('/')
def index():
    try:
        error = request.args['error']
    except:
        error = None
    onlyfiles = [f for f in listdir(UPLOAD_FOLDER) if isfile(join(UPLOAD_FOLDER, f))]
    onlyvid = [f for f in listdir(VIDEO_FOLDER) if isfile(join(VIDEO_FOLDER, f))]
    return render_template('index.html',lenF =len(onlyfiles),lenV=len(onlyvid),listfiles = onlyfiles,listvid = onlyvid,error=error)

@app.route('/uploader-lite', methods = ['GET', 'POST'])
def choose_file():
   if request.method == 'POST':
      select = request.form.get('comp_select')
      return redirect(url_for('chart',
                     filename=select))

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   e = None
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      if allowed_file(filename):
          f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          return redirect(url_for('chart',filename=f.filename))
      else :
          e ='Extension NOT ALLOWED'
          return redirect(url_for('index',error=e))

@app.route('/uploader-video', methods = ['GET', 'POST'])
def upload_vid():
   e = None
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      if allowed_vid(filename):
          #f.save(secure_filename(f.filename))
          #path = 'datasets/'+filename
          #dataF = pd.read_csv(path)
          f.save(os.path.join(app.config['VIDEO_FOLDER'], filename))
          #return redirect(url_for('chart',filename=f.filename))
      else :
          e ='Extension NOT ALLOWED'
      return redirect(url_for('index',error=e))

@app.route('/uploader-lite-video', methods = ['GET', 'POST'])
def choose_vid():
   if request.method == 'POST':
      nom = request.form.get('comp_select')
      path = 'Videos/'+nom
      cap = cv2.VideoCapture(path)
      length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      print( length )
      Response(gen(VideoCamera(path),nom[:-4],length), mimetype='multipart/x-mixed-replace; boundary=frame')
      return redirect(url_for('index',error=None))

@app.route('/Charts')
def chart():
    filename = request.args['filename']
    path = 'datasets/'+filename
    try:
        dataF = pd.read_csv(path)
    except Exception as e:
        return redirect(url_for('index',
                         error=e))
    dataF = dataF[['Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise','Contempt']]
    pie_Chart = pie_chart(dataF)
    radar = radar_chart(dataF)
    return render_template('charts.html',filename=filename,means = mean_data(dataF),df = df_to_arrays(dataF),pie=pie_chart(dataF),radar=radar)

@app.route('/panel-alert')
def panel():
    return render_template('panels.html')
'''
@app.route('/log')
def login():
    return render_template('login.html')
'''
if __name__ == '__main__':
    app.run()
