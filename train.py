import numpy
import sys
import os
import time
import cv2

size = 4
fn_haar = 'haarcascade_frontalface_alt.xml'
fn_dir = 'subject_faces'
fn_name = sys.argv[1]
subcounter = 7
num_pics = 25
# Text to lower case
fn_name = fn_name.lower()
# Ensures that there is a common folder to store subjects' folders
if not os.path.isdir(fn_dir):
    os.mkdir(fn_dir)
    print "Creating Pictures Database folder"
# Makes the subject's pictures folder
path = os.path.join(fn_dir, fn_name)
if os.path.isdir(path):
    print "Pulishing training for:" + fn_name
if not os.path.isdir(path):
    os.mkdir(path)
    print " Creating folder for new user named:" + fn_name
# Width and height of the picture
(im_width, im_height) = (224, 184)
haar_cascade = cv2.CascadeClassifier(fn_haar)
# Using webcam 0
webcam = cv2.VideoCapture(0)
# counter
count = 0
# waiting counter
wc = 0
# Number of photos that are going to be taken
while count < num_pics:

    # reading from webcam
    (rval, im) = webcam.read()
    # flipping image by Y axis
    im = cv2.flip(im, 1, 0)
    # changing image colors from BGR to Gray
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # resizing gray image 
    '''revisa esta parte jorge'''
    mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))
    # detacting for any scale
    faces = haar_cascade.detectMultiScale(mini)
    # sorting element by element based of faces
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        wc += 1
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        # conditioning luego le agrego un comentario coherente
        if wc >= subcounter:
            face_resize = cv2.resize(face, (im_width, im_height))
            # Looks down for the folder of the subject in training
            pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                          if n[0]!='.' ]+[0])[-1] + 1
            # Writes down the file
            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
            print "imagen capturada "+ str(count+1)
            wc = 0
            count += 1
        print wc
        
        if count >= 0 and count < 5:
            message = "Look forward"
        elif count >= 5 and count < 10:
            message = "Look slightly to the right"
        elif count >= 10 and count < 15:
            message = "Look slightly to the left"
        elif count >= 15 and count < 20:
            message = "Look slightly up"
        elif count >= 20 and count < 25:
            message = "Look slightly to the down"

          # Drawing the rectangle of the recognized face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Writes down subject Name
        cv2.putText(im,'Traning for: '+ fn_name, (x + 20, y - 20), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 0))
        cv2.putText(im, message , (x + 12, y - 8), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 0))
    # Writes down Window Title
    cv2.imshow('FaceIT - Training Mode | '+ fn_name, im)
    key = cv2.waitKey(10)
    if key == 27:
        break