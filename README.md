# Sentiment-Analysis-Adv

## General Architecture :
This project consists of two main parts, the first is about model training while the second part is about dashboard management and data visualization.
### Model training :
To train appropriate models to well visualize/Annalyse Sentiments, I tried 2 approaches : Baseline Model and Advanced ones.
#### Baseline Model :
It is a simple CNN, the convolutional part contains 4 convolutional layers, in addition, it uses the maximum function in the pooling part. On the other hand, there are two dense layers, the latter of which is activated by the softmax function. The inline text shows the general structure of the base form:
```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 48, 48, 64)        640       
_________________________________________________________________
batch_normalization_5 (Batch (None, 48, 48, 64)        256       
_________________________________________________________________
activation_5 (Activation)    (None, 48, 48, 64)        0         
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 24, 24, 64)        0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 24, 24, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 24, 24, 128)       73856     
_________________________________________________________________
batch_normalization_6 (Batch (None, 24, 24, 128)       512       
_________________________________________________________________
activation_6 (Activation)    (None, 24, 24, 128)       0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 12, 12, 128)       0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 12, 12, 128)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 12, 12, 512)       590336    
_________________________________________________________________
batch_normalization_7 (Batch (None, 12, 12, 512)       2048      
_________________________________________________________________
activation_7 (Activation)    (None, 12, 12, 512)       0         
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 6, 6, 512)         0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 6, 6, 512)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 6, 6, 512)         2359808   
_________________________________________________________________
batch_normalization_8 (Batch (None, 6, 6, 512)         2048      
_________________________________________________________________
activation_8 (Activation)    (None, 6, 6, 512)         0         
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 3, 3, 512)         0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 3, 3, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               1179904   
_________________________________________________________________
dense_2 (Dense)              (None, 512)               131584    
_________________________________________________________________
batch_normalization_9 (Batch (None, 512)               2048      
_________________________________________________________________
activation_9 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 3591      
=================================================================
Total params: 4,346,631
Trainable params: 4,343,175
Non-trainable params: 3,456
_________________________________________________________________
```
The database used here is ***Fer2013*** , it consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. The training set consists of 28,709 examples and the public test set consists of 3,589 examples. The model task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
To start the training, You have first to extract ***Fer2013.tar.xz***, here what you will find after extraction :
```bash
-rw-rw-r-- 1 mouad mouad 710530 شتنبر  11 09:11 Facial_Expression_Training.ipynb
-rw-rw-r-- 1 mouad mouad   6365 شتنبر  11 09:41 reqs.txt
drwxrwxr-x 9 mouad mouad   4096 مارس   13  2020 test
drwxrwxr-x 9 mouad mouad   4096 مارس   13  2020 train
drwxrwxr-x 4 mouad mouad   4096 يوليوز 14 11:51 utils
```
***train*** and ***test*** folders contain the training and validation data, ***utils*** contains functions used to support the training process, To start model training you better create new virtual environnemnt and install the requirements listed in ***reqs.txt**, the following command may make it easier :
```
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
python3 -m pip install -r reqs.txt
```
After that, you launch jupyter notebook to begin the training process.

#### Advanced Models :
After generating a baseline model, we will try to generate more advanced models and complicated ones, training them using more efficient Database: **AffectNet** eventhought we will use just a sample containing like 300000 colored 224x224 images annoated manually.
##### Database preprocessing:
The initial architecture of the database demand a sort of processing to make it coherent with Annalysis and predicting tools.To download the sample click [here](https://1drv.ms/u/s!Alvoz4G5IaBxiKo7ZQEM7cQMX5LCzg?e=d70fTL),It is about 4GB and there are two tar archive files that need to be downloaded (train_set.tar and val_set.tar); After extract the tar files, you will find the images and corresponding labels are stored in “images” and “annotations” folders. You extract both folders and apply the following instructions :
1. extract train_set.tar and val_set.tar into AffectNet folder.
2. prepare new conda environnement with packages mentionned on the **requirements.txt** file.(You may notice that I used conda environnement python3)
3. run ***genDataset.ipynb*** to prepare more efficient version of the training set.
4. run ***genDataset-Val.ipynb*** to prepare more efficient version of the validation set.

from now on we will use 2 newly generated directories, **dataset** for training set, **ValSet** for validation set, each one contains **8** sub-directories, each one of them present a unique sentiment reffered to it by a integer between 0 and 7 included. The following dictionnary define the attached emotion and its convenient code: 
> EMOTIONS_LIST = [ 'Neutral' : 0, 'Happy' : 1, 'Sad' : 2, 'Surprise' : 3, 'Fear' : 4, 'Disgust' : 5, 'Anger' : 6, 'Contempt' : 7

##### Model Training :
Contrary to the baseline model, here we gonna use line commands to personnalize/automate/personnalize the training process, the line command is :
```
python tee.py --model [VGG16, Resnet50, Xception, InceptionV3, MobileNetV2, MobileNet] --epochs [Number of epochs] --weights path/to/weights/file
```
This command permit to user to choose among 6 pre-built architectures - You can add any other architecture by personnalizing the code in **modelTrain.py** -  Also we may define epochs number and weights file if it has been already existed. The default value for epochs [weights] is 30 [None]

As result, 2 files will be added to models directory, the first has ".json" extension, it refers to the model architecture, the second ends by ".h5" it contains the saved weights after the training process.

### Dashboard Management :
Basically the created dashboard is a flask application, the choice between ** Dashboard-48 ** & ** Dashboard-224 ** depends on the model training strategy, the former requires a model generated based on '48x48 grayscale images [Fer2013] (), while the other is perfectly consistent with models trained using 224x224 colored images [AffectNet] (). And keep in mind that there are critical differences between the 2 dashboards *** So pay attention to the choice of model ***.

The first thing you do is to set a virtual python environnement based on the **requirements.txt** existing in the main directory along side with other sub directories. After activating the env, we run the following command :
```
python Dashboard-XX/app.py
```
The flask app will be launched on the localhost automatically, the main page will be like this:
![index](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/index.jpg)

![index](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/index2.jpg)

The first thing to do is to submit the wanted video, or if you have the full access to server, you may add it directly to **Videos** folder. Second, we choose the video from the second case, and we click on ***Submit***. Now the app will process the frames to generate in the end a csv dataset that summarizes for each frame the propotions of the different considered emotions depending on the used model.
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/chooseVid.png)

After generating the dataset, you must select the appropriate dataset and submit it to generate representative ***Charts*** and the final report.
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/chooseVid2.png)
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/chooseVid3.png)

> ET Voila !!!
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/chooseVid4.png)

#### Charts visualization :
```python
#app.py
def chart():
    filename = request.args['filename']
    path = 'datasets/'+filename
    try:
        dataF = pd.read_csv(path)
    except Exception as e:
        return redirect(url_for('index',
                         error=e))
    dataF = dataF[['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']]
    pie_Chart = pie_chart(dataF)
    radar = radar_chart(dataF)
    return render_template('charts.html',
                           filename=filename,means = mean_data(dataF),
                           df = df_to_arrays(dataF),
                           pie=pie_chart(dataF),
			   radar=radar)

```
```javascript
var data = {{df|tojson}};
```
For the visualization part, I choose to work with ***Plotly JS*** to easily maintain the coherence in term of efficency & convenience between ***Flask*** and the different automatically generated charts. in the comming parts, I will present charts generated from a random video that has been already submited.
##### EasypieChart :
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/charts5.png)
These mini-charts show the general mean of domination rates for each emotion in the video under study.
To get this result, we calculate columns means in the csv dataset that represents domination rates in our project
```python
#utils.py
def mean_data(df):
    arr ,L= df_to_arrays(df),[]
    for i in range(len(arr)):
        L.append(round(np.mean(arr[i]), 2))
        #print("%.2f" % np.mean(arr[i]))
    L = list(np.array(L))#//np.sum(np.array(L)) * 100)
    return [round(i,2) for i in L]
```
Then we define it while rendering ***charts.html*** in ***chart()*** method as you can see above. After that, we redefine it as json array in the template itself.
```javascript
//charts.html
var means ={{means|tojson}};
```
Finally To present Angry mini-chart -***for exemple*** -, we add the following block :
```html
<!--charts.html-->
<div class="col-xs-6 col-md-3">
 <div class="panel panel-default">
  <div class="panel-body easypiechart-panel">
   <h4>Angry</h4>
    <div class="easypiechart" id="easypiechart-teal" data-percent="{{means[0]}}"  >
	    <span class="percent">{{means[0]}}%</span>
    </div>
</div></div></div>
```
##### Line Chart :
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/charts1.png)

As plotly.JS documentation mentionnes, We gonna retreive the data through ***chart()*** method and then we add the following code to our html template to visualize all traces
```html
<div class="panel-body">
 <div id="myDiv8" class="canvas-wrapper">
 </div>
 <script>
  //['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
	var trace1 = {
               x: result,
               y: data[0],
               type: 'scatter',
	       name: 'Angry'
              };
	var trace2 = {
               x: result,
               y: data[1],
               type: 'scatter',
	       name: 'Disgust'
              };
	. . .					  
	var trace7 = {
	             x: result,
	             y: data[6],
	             type: 'scatter',
		     name: 'Surprise'
	             };
	var data = [trace1, trace2,trace3, trace4,trace5, trace6,trace7];
        Plotly.newPlot('myDiv8', data);
 </script>
</div>
```

##### Bar Chart :
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/charts2.png)

We will repeat the same process except here we have to adapt our code/data to plots we wish generate , for bar chart here is the code:
```html
<div class="panel-body">
 <div id="myDiv"></div>
  <script>
	pieC = {{pie|tojson}};
	var xValue = pieC[1];
	var yValue = pieC[0];
        var trace1 = {
	  x:xValue,
	  y:yValue,
	  type: 'bar',
          name: '',
          marker: {
               color: 'rgb(49,130,189)',
               opacity: 0.5
            }
          };
	var data = [trace1];

        var layout = {
          title: 'Significant Emotions',
          xaxis: {
          tickangle: -45
           },
          barmode: 'group'
          };

        Plotly.newPlot('myDiv', data, layout);
</script></div>
```
##### Radar Chart :
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/charts3.png)

The code generating Radar chart is :
```html
 <div class="canvas-wrapper">
  <div id="myDiv3"></div>
    <script>
	var radar = {{radar|tojson}};
		data = [{
			type: 'scatterpolar',
			r: radar[0],
			theta: radar[1],
			fill: 'toself'
			}]

		layout = {
			polar: {
			radialaxis: {
				visible: true,
				range: [0, 50]
				}
			},
			showlegend: false
		 }

	Plotly.newPlot("myDiv3", data, layout)
	//Plotly.newPlot('myDiv1', data, layout, config );
   </script>
</div>
```
##### Pie Chart :
![](https://github.com/RIALI-MOUAD/Summer-internship-/blob/main/charts4.png)

For Pie Chart we use :
```html
<div class="canvas-wrapper">
	<div id="myDiv2"></div>
		<script>
	    var pie = {{pie|tojson}}
            var data = [{
                values: pie[0],
                labels: pie[1],
                type: 'pie'
               }];

            var layout = {
               height: 400,
               width: 500
              };

            Plotly.newPlot('myDiv2', data)//, layout);
      </script>
</div>
```
