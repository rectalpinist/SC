import numpy as np
import cv2
from sklearn import datasets
from sklearn.svm import LinearSVC
from collections import Counter
from skimage.feature import hog

#Detect Line coordinates using Hough Transformationt
def detectLine(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edgesImg = cv2.Canny(grayImg, 50, 150, apertureSize = 3)

    minLineLength=200
    lines = cv2.HoughLinesP(image=edgesImg, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=20)

    a,b,c = lines.shape
   
    x1 = lines[0][0][0]
    y1 = 480 - lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = 480 - lines[0][0][3]
    
    return (x1, y1, x2, y2)

#Detect if number crossed the line
#Translirano za 5px dole-desno kako bi se izbegao "šum" linije
def detectCross(x, y, k, n):
    x = x-5
    y = y-5
    yy=k*x+n
    if (yy-y)<1.5 and (yy-y)>=0:
        return True
    else:
        return False



#Main
#Training the classifier based on MINST Original Dataset
#Kod za Obucavanje sa http://hanzratech.in
print "Ucitavam MINST Dataset..."

dataset = datasets.fetch_mldata("MNIST Original")
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

print "Pripremam skup..."

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()
clf.fit(hog_features, labels)

print "Trening gotov!"

#Prepare output file. Write header.
fileOut = open('out.txt', 'w')
fileOut.write("RA 168/2012 Igor Tot" + '\n' + "file" + '\t' + "sum" + '\n')
fileOut.close()

#Iterate trough videos and get their sums
for vidNum in np.arange(0, 10):
    #Initialize values
    suma = 0
    k = 0
    n = 0
    frameNum = 0;
    cap = cv2.VideoCapture("videos/video-" + str(vidNum) + ".avi")
    print "********** Obradjujem video-" + str(vidNum) + ".avi **********"
    cap.set(1, frameNum);
           
    while True:
        frameNum += 1
        ret, frame = cap.read()
        if not ret:
            break

        if (frameNum==1):
            lineCoords = detectLine(frame)
            lineLeftEdge = lineCoords[0]
            lineRightEdge = lineCoords[0]+lineCoords[2]            
            k = (float(lineCoords[3])-float(lineCoords[1]))/(float(lineCoords[2])-float(lineCoords[0]))
            n = k*(float(-lineCoords[0])) + float(lineCoords[1])
            print "**** Detektovana linija! ****"
            
        imgGray = np.ndarray((frame.shape[0], frame.shape[1]))
        for i in np.arange(0, frame.shape[0]):
            for j in np.arange(0, frame.shape[1]):
                if frame[i, j, 0] > 150 and frame[i, j, 1] > 150 and frame[i, j, 2] > 150:
                    imgGray[i, j] = 255
                else:
                    imgGray[i, j] = 0
        imgGray = imgGray.astype('uint8')
        imgNumber = cv2.dilate(imgGray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
        
        _, ctrs, _ = cv2.findContours(imgNumber.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        for rect in rects: 
            x = rect[0] + rect[2] / 2
            y = 480 - (rect[1] + rect[3] / 2)

            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            #kad je van ekrana error
            if (pt1<0):
                pt1=0
            if (pt2<0):
                pt2=0

            roi = imgNumber[pt1:pt1 + leng, pt2:pt2 + leng]
            #Resize
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_NEAREST)
            roi = cv2.dilate(roi, (3, 3))
            #Calculate HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

            if  lineLeftEdge <= x <= lineRightEdge and detectCross(x, y, k, n):
                suma += int(nbr)
                print 'Detektovan broj '+ str(nbr) + ' - frejm: ' + str(frameNum)

    print '**** Suma: ' + str(suma) + ' ****'
    cap.release()
    fileOut = open('out.txt', 'a')
    fileOut.write("video-" + str(vidNum) + ".avi" + '\t' + str(suma) + '\n')
    fileOut.close()