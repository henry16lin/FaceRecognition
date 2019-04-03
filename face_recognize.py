import cv2
import argparse
import numpy as np
import os
from scipy import misc
from PIL import Image

from keras.models import load_model


def recognize_img(cascName,classifier,model_name):
    a=0
    while True: ###local command mode
        a +=1
        imagePath = input('please input image path: ') #local command mode
        c = open('classes.txt','r')
        classes = c.read().split(',')
        #cascPath = "haarcascade_frontalface_default.xml"
    
        # Create the haar cascade
        cascPath = os.path.join(os.getcwd(),'haarcascade',cascName)
        faceCascade = cv2.CascadeClassifier(cascPath)
    
        # Read the image
        image = cv2.imread(imagePath)
        if image.size>3*10**7:
            image_resize = misc.imresize(image,0.33)  ## resize img to 1/3 original size
        else:
            image_resize = image
        gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
        
        try:
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
            	 #flags = cv2.CASCADE_SCALE_IMAGE
            )
            
            print("Found {0} objects!".format(len(faces)))
        except:
            print('**\n object locate fail...check cascade.xml file...\n**')   
            break
        
        
        if classifier:
            #load classifier model
            try:
                model_path_str = 'checkpoint/'+ model_name
                model = load_model(model_path_str)
                classifier = True
            except:
                print('**\n load model fail, only implement object located...check model in /checkpoint \n**')
                classifier = False
    
        
        for i in range(len(faces)):
            face = faces[i,]
            (x,y,w,h) = face
            
            if classifier:
                single_face = image_resize[ y:(y+int(np.around(w*1.214))), x:(x+w),:]
                single_face_resize = misc.imresize(single_face,[85,70,3])
                
                x_test4D = single_face_resize.reshape(1,85,70,3).astype('float32')
                x_test_normalize = x_test4D / 255
                
                class_prob = model.predict(x_test_normalize)[0]
                pred_class = np.argmax(class_prob)
      
                cv2.rectangle(image_resize, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = classes[pred_class]+'('+ str(np.around(class_prob[pred_class],3))+')'
                cv2.putText(image_resize, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            else:
                cv2.rectangle(image_resize, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = 'object'
                cv2.putText(image_resize, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        
        image_resize = image_resize[: , : , : : -1]
        img2 = Image.fromarray(image_resize, 'RGB')
        img2.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
            '--classifier', default=False, action="store_true",
            help='whether to apply model for classifing object'
        )
    parser.add_argument(
            '--haarcascade', type=str, default='haarcascade_frontalface_default.xml',
            help='opencv haarcascade xml file name'
        )
    parser.add_argument(
            '--model', type=str, default='model(BN+GAP+aug).h5',
            help='model weight file in checkpoint'
        )
    
    args = parser.parse_args()
    model_name = args.model
    cascName = args.haarcascade
    classifier = args.classifier
    
    recognize_img(cascName,classifier,model_name)
    
    
