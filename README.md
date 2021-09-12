# Sentiment-Analysis-Adv

## General Architecture :
This project has 2 basic parts, The first is related to model training while the second one is about Dashboard management and Data visualisation.
### Model training :
To train appropriate models to well visualize/Annalyse Sentiments, I tried 2 approaches : Baseline Model and Advanced ones.
#### Baseline Model :
It is a simple CNN, The convultional part has 4 Convultionnal layers, besides, I used maximum function in the pooling part. On the other hand, there is 2 dense layers, the last one is activated by softmax function. The embeded text shows the general structure of the baselone model:
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
***train*** and ***test*** folders contain the training and validation data, utils contains functions used to support the training process, To start model training you better create new virtual environnemnt and install the requirements listed in ***reqs.txt**, the following command may make it easier :
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
Basically, the dashboard created is a flask application, the choice between **Dashboard-48** & **Dashboard-224** depends on the model training strategy, the fisrt  requires model generated based 48x48 grayscale images [Fer2013](),While the other is perfectly coherent with models trained using 224x224 colored images [AffectNet](). And keep in mind that there is some critical differences between the 2 dashboards ***So pay attention to the model choice***.

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
