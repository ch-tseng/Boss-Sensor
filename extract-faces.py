import cv2
import glob
import os

sourePath = "datasets\\photos"
savePath = "datasets\\faces"
face_size_min = (47, 62)
cascade = "lbpcascade_frontalface.xml"  # lbpcascade_frontalface.xml / haarcascade_frontalface_default.xml

def checkDir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

checkDir(savePath)
	
face_cascade = cv2.CascadeClassifier('objects\\' + cascade)

i = 0

for filePath in glob.glob(sourePath+"\\*"):
    print("Load {} ...".format(filePath))
    
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)

    for (x,y,w,h) in faces:
        if(w>=face_size_min[0] and h>=face_size_min[1]):
            roi_color = img[y:y+h, x:x+w]
            cv2.imwrite(savePath+"/face-"+str(i)+".jpg", roi_color)
            i+=1
			
print("All faces has been extracted to {}".format(savePath))
