# FaceRecognition (cv2 Recognition)  
openCV + classifier for 2-stage object detection
 
cv2 4.0.0  
tensorflow 1.12.0  
keras 2.2.4  

## Quick Start:  
1. open command window and cd to your path  
2. *`python face_recognize.py`*  
3. input the image path:  
    ex: *C:\pythonworks\FaceRecognition\people.jpg*

## Usage:  
Use --help to see usage of face_recognize.py:  
```
usage: face_recognize.py [-h] [--classifier] [--haarcascade HAARCASCADE]  
                         [--model MODEL]     
                           
optional arguments:  
  -h, --help            show this help message and exit  
  --classifier          whether to apply model for classifing object  
  --haarcascade HAARCASCADE opencv haarcascade xml file name  
  --model MODEL         model weight file in checkpoint  


--haarcascade: opencv haarcascade xml file name. default setting is 'haarcascade_frontalface_default.xml'  
--classifier: use classifer to classify object catch in openCV  
--model: model name in /checkpoint(keras model. if want to change to other type of model, some change is needed in face_recognize.py)  
```  

classifier example: *`python face_recognize.py --haarcascade haarcascade_frontalcatface.xml --classifer --model your_model_name.h5`*   


# training:
You can use `train.py` to train your own model with structure in `model_structure.py` from random initial  
or use `train_by_transfer_learning.py` to do transfer learning from VGG16 which trained weight from imagenet.  
  
Put your training date in training data folder and different class in different folder  
than modify `classes.txt` to your class name which split by ','



