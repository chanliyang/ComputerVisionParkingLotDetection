# Parking Lot Occupancy Detection
This project implements four different methods to detect parking lot occupancy (Empty vs. Occupied) using computer vision techniques on the PKLot dataset. The methods include Traditional image processing, Perceptual Hashing (pHash), Support Vector Machine (SVM) with HOG features, and a modified AlexNet (mAlexNet) deep learning model. The code is written in Python and uses libraries such as OpenCV, scikit-learn, TensorFlow, and others.

## Project Overview
The goal is to classify parking spaces as "Empty" (0) or "Occupied" (1) based on images from the PKLot dataset. The dataset contains images of parking lots with COCO-format annotations, and the code processes these images to extract features and make predictions. Each method is evaluated using accuracy, precision, recall, F1-score, and confusion matrices, with visualizations of preprocessing steps provided for better understanding.

### Methods
1. **Traditional Method**: Uses Otsu's thresholding and morphological operations to segment parking spaces and classify based on connected component areas.
2. **Perceptual Hashing (pHash)**: Computes perceptual and difference hashes, comparing them against reference hashes to classify parking spaces.
3. **SVM with HOG Features**: Extracts Histogram of Oriented Gradients (HOG) features and uses a pre-trained SVM model for classification.
4. **mAlexNet**: A modified AlexNet convolutional neural network (CNN) that processes cropped and normalized images, using a pre-trained model for classification.

## Dataset
The project uses the PKLot dataset, specifically the `lot2` subset with images from December 7, 2012, to December 31, 2012. The dataset is organized into:
- **Train**: Training images and annotations (`_annotations.coco.json`).
- **Valid**: Validation images and annotations.
- **Test**: Test images and annotations.

## Sample Output 
<img width="801" height="1572" alt="image" src="https://github.com/user-attachments/assets/4c4b9497-8842-4c26-9cfa-e4b794777ce3" />
<img width="658" height="545" alt="image" src="https://github.com/user-attachments/assets/257d99b1-1876-4196-8241-205da1dc30db" />
