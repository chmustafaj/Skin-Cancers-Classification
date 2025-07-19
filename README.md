{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Skin Cancers Classification\
\
\
## Project Objective\
\
The goal of this project was to **develop a framework** capable of:\
- Segmenting all **12 types of tissue pixels** in medical images.\
- Classifying an image **patch into one of three classes**.\
\
This required integrating knowledge from Digital Image Processing, Neural Networks, Programming, and Mathematics.\
\
---\
\
## Methodology\
\
### 1. **Segmentation**\
- **Architecture**: A neural network based on the **U-Net** architecture.\
- **Dataset Split**: 90% training / 10% testing.\
- **Preprocessing**:\
  - Converted class colors to integer indexes.\
- **Training**:\
  - Trained for 150 epochs.\
- **Output**:\
  - Accuracy checked using confusion matrix.\
  - Predicted masks saved in the `Outputs` folder.\
\
### 2. **Feature Extraction**\
- **Features**:\
  - **12 areas** and **12 perimeters** for each tissue type (24 features total).\
- **Process**:\
  - Count pixels of each class for areas.\
  - Convert colored image into binary masks (one per class) for perimeter extraction.\
  - Find contours of each mask to compute perimeters.\
\
### 3. **Classification**\
- **Classifier**: Euclidean distance-based.\
- **Approach**:\
  - Compute mean feature vector for each class.\
  - Classify test image based on smallest distance to mean vector.\
\
---\
\
## Results\
\
- **Classification Accuracy**: 43.33%\
- **Evaluation Metric**: Confusion Matrix\
- **Tested on**: Provided test dataset\
\
---\
\
## Output Samples\
\
> _Sample outputs from the segmentation process were saved to the `Outputs/` folder (not included here)._\
\
---\
\
## Flow Diagram\
\
> _Refer to the project PDF for the visual flow diagram._\
\
\
}