import tensorflow 
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Live inferece for mask detection')
parser.add_argument('--camera',type=int,help='camera device number')
parser.add_argument('--picture-mode',type=int,help='picture mode: supply 1 to run')
parser.add_argument('--source',type=str,help='image path')


kwargs = vars(parser.parse_args())


MASK_MODEL_PATH = os.getcwd() + '/model_facemask.h5'
mask_model = load_model(MASK_MODEL_PATH)
haarcascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
inference ={ 
    0:{
        'result':'maskless',
        'color':(0,255,0)
    },
    1:{
        'result':'mask',
        'color':(255,0,0)
    }
}
config = {
    'bbox_size':4
}


def detect_mask(image):
    bbox_size = config['bbox_size']
    resize = cv2.resize(
        image, 
        (image.shape[1] // bbox_size,
            image.shape[0] // bbox_size)
    )

    # Get x,y and weight and height of detected faces 
    faces = haarcascade.detectMultiScale(resize)

    #for each face, crop the faces and pass to a mask detector model and draw bbox
    for f in faces:
        (x, y, w, h) = list(np.array(f)*bbox_size)

        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img,(150,150))/225.0
        face_img =  np.vstack([
            np.reshape(face_img,(1,150,150,3))
        ])
        
        prediction_probability = mask_model.predict(face_img)
        target_class = np.argmax(prediction_probability,axis=1)[0]

        
        #draw box and prediction class on image
        cv2.rectangle(
            image,(x,y),(x+w,y+h),
            inference[target_class]['color'],2
        )
        cv2.rectangle(
            image,(x,y-40),(x+w,y),
            inference[target_class]['color'],-1)
        cv2.putText(image, inference[target_class]['result'], 
                    (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(255,255,255),
                    2)
    return image


print(kwargs)
if kwargs.get('picture_mode'):
    if kwargs.get('picture_mode') == 1:
        if kwargs.get('source'):
            try:
                image = cv2.imread(kwargs['source'])
                infered = detect_mask(image)
                cv2.imwrite('out.png',infered)
            except Exception as e:
                print(f"No such image: {kwargs['source']} found ")
                print(e)
        else:
            print("None image path provided")
    else:
        import sys
        sys.exit()

else:

    device = kwargs['camera'] if kwargs['camera'] else 0 

    cap = cv2.VideoCapture(device) 

    while True:

        # Read image from video capturer
        try:
            _,image = cap.read()
        

            image = cv2.flip(image,1,1) 

        except:
            print('Such video device is not available')
            break

        image = detect_mask(image)

        cv2.imshow('CAMERA',  image)
        cv2.waitKey(10)
        
        key = cv2.waitKey(10)
        if key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()