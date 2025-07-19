import cv2
import numpy as np
from PIL import Image
import os
from sklearn.utils import shuffle

def findPerimeters(image):
    images = divideSegmentsToGrayscale(image)
    perimeters = []
    for image in images:
        # Find contours in the binary image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate over the contours and calculate the perimeters
        perimeterOfTissue = 0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            perimeterOfTissue+=perimeter
        perimeters.append(perimeterOfTissue)  
    return np.array(perimeters).astype(int)        

def divideSegmentsToGrayscale(colorImage):
    GLD = np.zeros([256,256], dtype=np.uint8)
    INF = np.zeros([256,256], dtype=np.uint8)
    FOL = np.zeros([256,256], dtype=np.uint8)
    HYP = np.zeros([256,256], dtype=np.uint8)
    RET = np.zeros([256,256], dtype=np.uint8)
    PAP = np.zeros([256,256], dtype=np.uint8)
    EPI = np.zeros([256,256], dtype=np.uint8)
    KER = np.zeros([256,256], dtype=np.uint8)
    BKG = np.zeros([256,256], dtype=np.uint8)
    BCC = np.zeros([256,256], dtype=np.uint8)
    SCC = np.zeros([256,256], dtype=np.uint8)
    IEC = np.zeros([256,256], dtype=np.uint8)

    for i in range(colorImage.shape[0]):
        for j in range(colorImage.shape[1]):
            pixelColor = colorImage[i,j]
            if np.array_equal(pixelColor, [115, 0, 108]):
                GLD[j][i] = 255
            elif np.array_equal(pixelColor, [122, 1, 145]):
                INF[j][i] = 255
            elif np.array_equal(pixelColor, [148, 47, 216]):
                FOL[j][i] = 255
            elif np.array_equal(pixelColor, [242, 246, 254]):
                HYP[j][i] = 255
            elif np.array_equal(pixelColor, [130, 9, 181]):
                RET[j][i] = 255
            elif np.array_equal(pixelColor, [157, 85, 236]):
                PAP[j][i] = 255
            elif np.array_equal(pixelColor, [106, 0, 73]):
                EPI[j][i] = 255
            elif np.array_equal(pixelColor, [168, 123, 248]):
                KER[j][i] = 255
            elif np.array_equal(pixelColor, [0, 0, 0]):
                BKG[j][i] = 255
            elif np.array_equal(pixelColor, [255, 255, 127]):
                BCC[j][i] = 255
            elif np.array_equal(pixelColor, [142, 255, 127]):
                SCC[j][i] = 255
            elif np.array_equal(pixelColor, [127, 127, 255]):
                IEC[j][i] = 255
    return [GLD, INF,FOL,HYP, RET, PAP, EPI, KER, BKG, BCC, SCC, IEC]               

def findAreas(image):

    Gland_counters =0
    inflammation_counter=0
    Hair_counter=0
    Hypodermic_counter=0
    Reticular_counter=0
    papillary_counter=0
    Epidermic_counter=0
    Kertain_counter=0
    Background_counter=0
    Basal_counter=0
    Squamous_counter=0
    intra_counter=0
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            pixelColor = image[j][k]
    
            if np.array_equal(pixelColor, [115, 0, 108]):
                Gland_counters += 1
            elif np.array_equal(pixelColor, [122, 1, 145]):
                inflammation_counter += 1
            elif np.array_equal(pixelColor, [148, 47, 216]):
                Hair_counter += 1
            elif np.array_equal(pixelColor, [242, 246, 254]):
                Hypodermic_counter += 1
            elif np.array_equal(pixelColor, [130, 9, 181]):
                Reticular_counter += 1
            elif np.array_equal(pixelColor, [157, 85, 236]):
                papillary_counter += 1
            elif np.array_equal(pixelColor, [106, 0, 73]):
                Epidermic_counter += 1
            elif np.array_equal(pixelColor, [168, 123, 248]):
                Kertain_counter += 1
            elif np.array_equal(pixelColor, [0, 0, 0]):
                Background_counter += 1
            elif np.array_equal(pixelColor, [255, 255, 127]):
                Basal_counter += 1
            elif np.array_equal(pixelColor, [142, 255, 127]):
                Squamous_counter += 1
            elif np.array_equal(pixelColor, [127, 127, 255]):
                intra_counter += 1
                    
    print([Gland_counters,inflammation_counter,Hair_counter,Hypodermic_counter,Reticular_counter,papillary_counter,Epidermic_counter,Kertain_counter,Background_counter,Basal_counter,Squamous_counter,intra_counter] )
    return [Gland_counters,inflammation_counter,Hair_counter,Hypodermic_counter,Reticular_counter,papillary_counter,Epidermic_counter,Kertain_counter,Background_counter,Basal_counter,Squamous_counter,intra_counter]   

# Reading training folder
dir = './Output/Train'
a=os.listdir(dir)
counter = 0
labelsCounter = 0
TrainingArray = np.zeros([1350,256,256,3], dtype=np.uint8)
labels = np.zeros(len(TrainingArray), dtype=np.uint8)
for i in range(1, len(a)):
    currentFolder=os.listdir(os.path.join(dir, a[i]))
    for j in range(len(currentFolder)):
        img = cv2.imread((os.path.join(dir, a[i], currentFolder[j])), 1)
        TrainingArray[counter]= cv2.resize(img, (256,256))
        counter+=1
        labels[labelsCounter] = i
        labelsCounter+=1
TrainingLabelsVector = np.zeros([1350,2], dtype=np.uint8)
for i in range(len(TrainingLabelsVector)):
    if(labels[i]==1):
        TrainingLabelsVector[i] = [0,1]    # BCC
    elif(labels[i]==2):
        TrainingLabelsVector[i] = [1,0]    # IEC
    elif(labels[i]==3):
        TrainingLabelsVector[i] = [1,1]    #SCC

training_files, training_labels = shuffle(TrainingArray, TrainingLabelsVector)

areasBCC = []
areasIEC = []
areasSCC = []
# Finding the areas, and appending the vector with all the areas for each tissue to the corrosponsing class matrix
for i in range(len(training_files)): 
    print(i)
    imageAreas = findAreas(training_files[i])
    if np.array_equal(training_labels[i], [0, 1]):
        areasBCC.append(imageAreas)
    elif np.array_equal(training_labels[i], [1, 0]):
        areasIEC.append(imageAreas)
    elif np.array_equal(training_labels[i], [1, 1]):
        areasSCC.append(imageAreas)
np.save('areas_BCC.npy',areasBCC)
np.save('areas_IEC.npy',areasIEC)                    
np.save('areas_SCC.npy',areasSCC)                    


perimetersBCC = []
perimetersSCC = []
perimetersIEC = []
for i in range(len(training_files)): 
    print(i)
    imagePerimeters = findPerimeters(training_files[i])
    if np.array_equal(training_labels[i], [0, 1]):
        perimetersBCC.append(imagePerimeters)
    elif np.array_equal(training_labels[i], [1, 0]):
        perimetersIEC.append(imagePerimeters)
    elif np.array_equal(training_labels[i], [1, 1]):
        perimetersSCC.append(imagePerimeters)

np.save('perimeters_BCC.npy',perimetersBCC)
np.save('perimeters_IEC.npy',perimetersIEC)                    
np.save('perimeters_SCC.npy',perimetersSCC)