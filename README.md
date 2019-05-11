# Object Recognition  
cv2(haarcascade) + classifier for 2-stage object detection
 
cv2 4.0.0  
tensorflow 1.12.0  
keras 2.2.4  

## Quick Start:  
1. open command window and cd to your path  
2. *`python recognize.py`*  
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

classifier example: *`python recognize.py --haarcascade haarcascade_frontalcatface.xml --classifer --model your_model_name`*   


# training:
You can use `train.py` to train your own model with structure in `model_structure.py` from random initial weight  
or use `train_by_transfer_learning.py` to do transfer learning from pre-trained model in keras.  
  
Put your training data in folder `training data` and each class correspond to each sub-folder  
than modify `classes.txt` to your class name which split by ','



