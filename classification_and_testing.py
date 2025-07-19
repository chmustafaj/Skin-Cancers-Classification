import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
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


def test(testPatch):
    distanceFromBCC = np.linalg.norm(testPatch - meanVectorBCC)
    distanceFromIEC = np.linalg.norm(testPatch - meanVectorIEC)
    distanceFromSCC = np.linalg.norm(testPatch - meanVectorSCC)

    if(np.min(np.array([distanceFromBCC, distanceFromIEC, distanceFromSCC]))==distanceFromBCC):
        return 'BCC'
    elif(np.min(np.array([distanceFromBCC, distanceFromIEC, distanceFromSCC]))==distanceFromIEC):
        return 'IEC'
    elif(np.min(np.array([distanceFromBCC, distanceFromIEC, distanceFromSCC]))==distanceFromSCC):
        return 'SCC'
# Loading the fearture matrices
areasBCC = np.load('./areas_BCC.npy')
areasIEC = np.load('./areas_IEC.npy')
areasSCC = np.load('./areas_SCC.npy')

perimetersBCC = np.load('./perimeters_BCC.npy')
perimetersIEC = np.load('./perimeters_IEC.npy')
perimetersSCC = np.load('./perimeters_SCC.npy')

meanVectorBCC = []
meanVectorIEC = []
meanVectorSCC = []

# Split the matrix into 12 horizontal vectors so we can find the mean vectors
areasForEachTissueBCC = np.hsplit(areasBCC, 12)
areasForEachTissueIEC = np.hsplit(areasIEC, 12)
areasForEachTissueSCC = np.hsplit(areasSCC, 12)

perimetersForEachTissueBCC = np.hsplit(perimetersBCC, 12)
perimetersForEachTissueIEC = np.hsplit(perimetersIEC, 12)
perimetersForEachTissueSCC = np.hsplit(perimetersSCC, 12)

for areas in areasForEachTissueBCC:
    meanVectorBCC.append(np.mean(areas))
for perimeters in perimetersForEachTissueBCC:
    meanVectorBCC.append(np.mean(perimeters))   

for areas in areasForEachTissueIEC:
    meanVectorIEC.append(np.mean(areas))
for perimeters in perimetersForEachTissueIEC:
    meanVectorIEC.append(np.mean(perimeters))  

for areas in areasForEachTissueSCC:
    meanVectorSCC.append(np.mean(areas))
for perimeters in perimetersForEachTissueSCC:
    meanVectorSCC.append(np.mean(perimeters))    
  
# Reading the testing folder  
dir = './Output/Test'
a=os.listdir(dir)
counter = 0
labelsCounter = 0
TestingImages = np.zeros([150, 256,256,3], dtype=np.uint8)
labels = np.zeros(len(TestingImages), dtype=np.uint8)
for i in range(1, len(a)):
    currentFolder=os.listdir(os.path.join(dir, a[i]))
    for j in range(len(currentFolder)):
        img = cv2.imread((os.path.join(dir, a[i], currentFolder[j])), 1)
        TestingImages[counter]= cv2.resize(img, (256,256))
        counter+=1
        labels[labelsCounter] = i
        labelsCounter+=1

TestinglabelsVector = np.zeros([150,2], dtype=np.uint8)
for i in range(len(TestinglabelsVector)):
    if(labels[i]==1):
        TestinglabelsVector[i] = [0,1]
    elif(labels[i]==2):
        TestinglabelsVector[i] = [1,0]
    elif(labels[i]==3):
        TestinglabelsVector[i] = [1,1]
testing_files, testing_labels = shuffle(TestingImages, TestinglabelsVector)

# Fidning the features of the images in the testing folder
testingAreasBCC = []
testingAreasIEC = []
testingAreasSCC = []
for i in range(len(testing_files)): 
    print(i)
    imageAreas = findAreas(testing_files[i])
    if np.array_equal(testing_labels[i], [0, 1]):
        testingAreasBCC.append(imageAreas)
    elif np.array_equal(testing_labels[i], [1, 0]):
        testingAreasIEC.append(imageAreas)
    elif np.array_equal(testing_labels[i], [1, 1]):
        testingAreasSCC.append(imageAreas)
                   

testingPerimetersBCC = []
testingPerimetersSCC = []
testingPerimetersIEC = []
for i in range(len(testing_files)): 
    print(i)
    imagePerimeters = findPerimeters(testing_files[i])
    if np.array_equal(testing_labels[i], [0, 1]):
        testingPerimetersBCC.append(imagePerimeters)
    elif np.array_equal(testing_labels[i], [1, 0]):
        testingPerimetersIEC.append(imagePerimeters)
    elif np.array_equal(testing_labels[i], [1, 1]):
        testingPerimetersSCC.append(imagePerimeters)
np.save('testing_areas_BCC.mpy',testingAreasBCC)       
np.save('testing_areas_IEC.mpy',testingAreasIEC)       
np.save('testing_areas_SCC.mpy',testingAreasSCC)       

testingFeaturesBCC = np.hstack((testingAreasBCC, testingPerimetersBCC))
testingFeaturesIEC = np.hstack((testingAreasIEC, testingPerimetersIEC))
testingFeaturesSCC = np.hstack((testingAreasSCC, testingPerimetersSCC))

np.save('testing_features_BCC', testingFeaturesBCC)
np.save('testing_features_IEC', testingFeaturesIEC)
np.save('testing_features_SCC', testingFeaturesSCC)

# Putting the feature vectors together
allTestingFeatures = np.vstack((testingFeaturesBCC, testingFeaturesIEC, testingFeaturesSCC))
y_actual = []
y_pred = []
# Initializing the array with the actual labels
for i in range(len(allTestingFeatures)):
    patchType = test(allTestingFeatures[i])
    y_pred.append(patchType)
for i in range(len(testingFeaturesBCC)):
    y_actual.append('BCC')
for i in range(len(testingFeaturesIEC)):
    y_actual.append('IEC')
for i in range(len(testingFeaturesSCC)):
    y_actual.append('SCC')   
# Finding the accuracy and plotting the confiusion matrix  
accuracy = accuracy_score(y_actual, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_actual, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BCC', 'IEC', 'SCC'])
display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
