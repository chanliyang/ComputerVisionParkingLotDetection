# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:33:51 2025

@author: Raymo
"""

import os
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import imagehash
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Paths
BASE_PATH = r"C:\Users\Raymo\OneDrive\Desktop\Sunway Uni\Classes\sem 9\Computer Vision\Assignment\AssignmentCode\Dataset\PKlot"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VALID_PATH = os.path.join(BASE_PATH, "valid")
TEST_PATH = os.path.join(BASE_PATH, "test")
TRAIN_ANNOTATION = os.path.join(TRAIN_PATH, "_annotations.coco.json")
VALID_ANNOTATION = os.path.join(VALID_PATH, "_annotations.coco.json")
TEST_ANNOTATION = os.path.join(TEST_PATH, "_annotations.coco.json")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Constants
IMG_SIZE = (64, 64)
IMG_SIZE_PHASH = (128, 128)
PARKING_LOT = 'lot2'
LOT_DATES = (datetime(2012, 12, 7), datetime(2012, 12, 31))
AREA_THRESHOLD = 0.275
MAX_REFERENCES = 20
HAMMING_THRESHOLD = 20

# Load COCO annotations
def load_coco_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    image_map = {}
    for img in coco_data['images']:
        try:
            date_str = img['file_name'].split('_')[0]
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, IndexError):
            date = None
        image_map[img['id']] = {'file_name': img['file_name'], 'date': date}
    
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        label = 0 if ann['category_id'] == 1 else 1 if ann['category_id'] == 2 else None
        if label is not None:
            annotations[img_id].append({'bbox': ann['bbox'], 'label': label})
    return image_map, annotations







#--------------------------------------------Method 1------------------------------------------------------
#--------------------------------------------Traditional------------------------------------------------------
# Traditional Method
def preprocess_image_traditional(image, bbox, target_size=IMG_SIZE):
    x, y, w, h = map(int, bbox)
    cropped = image[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return cleaned, resized, gray, binary, opened

def load_dataset_traditional_visualize(data_path, annotation_path, date_range=None):
    image_map, annotations = load_coco_annotations(annotation_path)
    sample_images, sample_gray_images, sample_binary_images, sample_opened_images, sample_cleaned_images = [], [], [], [], []
    empty_count, occupied_count = 0, 0
    
    for img_id, img_info in image_map.items():
        if date_range and img_info['date']:
            if not (date_range[0] <= img_info['date'] <= date_range[1]):
                continue
        img_path = os.path.join(data_path, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        for ann in annotations.get(img_id, []):
            cleaned_img, resized_img, gray_img, binary_orig, opened_img = preprocess_image_traditional(image, ann['bbox'])
            if ann['label'] == 0 and empty_count < 1:
                sample_images.append((resized_img, ann['label']))
                sample_gray_images.append(gray_img)
                sample_binary_images.append(binary_orig)
                sample_opened_images.append(opened_img)
                sample_cleaned_images.append(cleaned_img)
                empty_count += 1
            elif ann['label'] == 1 and occupied_count < 1:
                sample_images.append((resized_img, ann['label']))
                sample_gray_images.append(gray_img)
                sample_binary_images.append(binary_orig)
                sample_opened_images.append(opened_img)
                sample_cleaned_images.append(cleaned_img)
                occupied_count += 1
            if empty_count >= 1 and occupied_count >= 1:
                break
        if empty_count >= 1 and occupied_count >= 1:
            break
    
    return sample_images, sample_gray_images, sample_binary_images, sample_opened_images, sample_cleaned_images

def load_dataset_traditional_test(data_path, annotation_path, date_range=None):
    image_map, annotations = load_coco_annotations(annotation_path)
    labels = []
    
    for img_id, img_info in image_map.items():
        if date_range and img_info['date']:
            if not (date_range[0] <= img_info['date'] <= date_range[1]):
                continue
        img_path = os.path.join(data_path, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        for ann in annotations.get(img_id, []):
            cleaned_img, _, _, _, _ = preprocess_image_traditional(image, ann['bbox'])
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned_img)
            total_area = sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels))
            img_area = cleaned_img.size
            area_ratio = total_area / img_area
            predicted_label = 1 if area_ratio > AREA_THRESHOLD else 0
            labels.append((ann['label'], predicted_label))
    
    true_labels = [lbl[0] for lbl in labels]
    pred_labels = [lbl[1] for lbl in labels]
    return np.array(true_labels), np.array(pred_labels)

def plot_samples_traditional(sample_images, sample_gray_images, sample_binary_images, sample_opened_images, sample_cleaned_images):
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    titles = ['Original', 'Grayscale', 'Otsu Threshold', 'Morphological Opening', 'Morphological Closing']
    
    for i, ((img, label), gray_img, binary_img, opened_img, cleaned_img) in enumerate(zip(sample_images, sample_gray_images, sample_binary_images, sample_opened_images, sample_cleaned_images)):
        images = [img, gray_img, binary_img, opened_img, cleaned_img]
        for j, (image, title) in enumerate(zip(images, titles)):
            if j == 0:
                axes[j, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[j, i].imshow(image, cmap='gray')
            axes[j, i].set_title(f"{title} {'Occupied' if label == 1 else 'Empty'}")
            axes[j, i].axis('off')
    
    plt.suptitle("Preprocessing Steps for Traditional Method", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def traditional_method():
    train_samples, train_gray, train_binary, train_opened, train_cleaned = load_dataset_traditional_visualize(
        TRAIN_PATH, TRAIN_ANNOTATION, LOT_DATES)
    
    test_true_labels, test_pred_labels = load_dataset_traditional_test(
        TEST_PATH, TEST_ANNOTATION, LOT_DATES)
    
    plot_samples_traditional(train_samples, train_gray, train_binary, train_opened, train_cleaned)
    
    accuracy = accuracy_score(test_true_labels, test_pred_labels)
    precision = precision_score(test_true_labels, test_pred_labels)
    recall = recall_score(test_true_labels, test_pred_labels)
    f1 = f1_score(test_true_labels, test_pred_labels)
    
    print(f"Traditional Method Accuracy: {accuracy:.4f}")
    print(f"Traditional Method Precision: {precision:.4f}")
    print(f"Traditional Method Recall: {recall:.4f}")
    print(f"Traditional Method F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(test_true_labels, test_pred_labels, "Traditional Method")
    
    return accuracy





#--------------------------------------------Method 2------------------------------------------------------
#--------------------------------------------pHash------------------------------------------------------
# Perceptual Hashing Method
def preprocess_image_phash(image, bbox, target_size=IMG_SIZE_PHASH):
    x, y, w, h = map(int, bbox)
    cropped = image[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    pil_image = Image.fromarray(thresh)
    phash = imagehash.phash(pil_image)
    dhash = imagehash.dhash(pil_image)
    return (phash, dhash), resized, gray, equalized, thresh

def select_reference_hashes(hashes, labels, max_refs=MAX_REFERENCES):
    empty_hashes = [h for h, l in zip(hashes, labels) if l == 0]
    occupied_hashes = [h for h, l in zip(hashes, labels) if l == 1]
    
    def cluster_hashes(hash_list, n_clusters):
        if not hash_list or len(hash_list) < n_clusters:
            return hash_list
        hash_vectors = np.array([[int(b) for b in h[0].hash.flatten()] for h in hash_list])
        kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, len(hash_list)), random_state=42)
        kmeans.fit(hash_vectors)
        centers = kmeans.cluster_centers_
        selected = []
        for center in centers:
            distances = [np.sum((hash_vectors[i] - center) ** 2) for i in range(len(hash_vectors))]
            selected.append(hash_list[np.argmin(distances)])
        return selected

    empty_refs = cluster_hashes(empty_hashes, min(max_refs, len(empty_hashes)))
    occupied_refs = cluster_hashes(occupied_hashes, min(max_refs, len(occupied_hashes)))
    return {'empty': empty_refs, 'occupied': occupied_refs}

def load_dataset_phash(data_path, annotation_path, date_range=None, max_refs=MAX_REFERENCES):
    image_map, annotations = load_coco_annotations(annotation_path)
    hashes, labels, sample_images, sample_gray_images, sample_equalized_images, sample_thresh_images = [], [], [], [], [], []
    empty_count, occupied_count = 0, 0
    
    for img_id, img_info in image_map.items():
        if date_range and img_info['date']:
            if not (date_range[0] <= img_info['date'] <= date_range[1]):
                continue
        img_path = os.path.join(data_path, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        for ann in annotations.get(img_id, []):
            hash_pair, resized_img, gray_img, equalized_img, thresh_img = preprocess_image_phash(image, ann['bbox'])
            hashes.append(hash_pair)
            labels.append(ann['label'])
            if ann['label'] == 0 and empty_count < 1:
                sample_images.append((resized_img, ann['label']))
                sample_gray_images.append(gray_img)
                sample_equalized_images.append(equalized_img)
                sample_thresh_images.append(thresh_img)
                empty_count += 1
            elif ann['label'] == 1 and occupied_count < 1:
                sample_images.append((resized_img, ann['label']))
                sample_gray_images.append(gray_img)
                sample_equalized_images.append(equalized_img)
                sample_thresh_images.append(thresh_img)
                occupied_count += 1
    
    reference_hashes = select_reference_hashes(hashes, labels, max_refs)
    return np.array(hashes, dtype=object), np.array(labels), sample_images, reference_hashes, sample_gray_images, sample_equalized_images, sample_thresh_images

def classify_phash(test_hash, reference_hashes, threshold=HAMMING_THRESHOLD):
    def compute_score(hashes, ref_list):
        if not ref_list:
            return float('inf')
        scores = [(0.6 * (test_hash[0] - ref[0]) + 0.4 * (test_hash[1] - ref[1])) for ref in ref_list]
        weights = [1 / (s + 1e-6) for s in scores]
        total_weight = sum(weights)
        if total_weight == 0:
            return float('inf')
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    empty_score = compute_score(test_hash, reference_hashes['empty'])
    occupied_score = compute_score(test_hash, reference_hashes['occupied'])
    
    if empty_score < occupied_score and empty_score < threshold:
        return 0
    elif occupied_score <= empty_score and occupied_score < threshold:
        return 1
    else:
        return 0 if empty_score < occupied_score else 1

def tune_threshold(val_hashes, val_labels, reference_hashes):
    thresholds = range(5, 50, 2)
    best_threshold, best_accuracy = HAMMING_THRESHOLD, 0
    for thresh in thresholds:
        val_pred = [classify_phash(h, reference_hashes, thresh) for h in val_hashes]
        acc = accuracy_score(val_labels, val_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thresh
    return best_threshold

def plot_samples_phash(sample_images, sample_gray_images, sample_equalized_images, sample_thresh_images):
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    titles = ['Original', 'Grayscale', 'Histogram Equalization', 'Adaptive Threshold']
    for i, ((img, label), gray_img, equalized_img, thresh_img) in enumerate(zip(sample_images, sample_gray_images, sample_equalized_images, sample_thresh_images)):
        images = [img, gray_img, equalized_img, thresh_img]
        for j, (image, title) in enumerate(zip(images, titles)):
            if j == 0:
                axes[j, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[j, i].imshow(image, cmap='gray')
            axes[j, i].set_title(f"{title} {'Occupied' if label == 1 else 'Empty'}")
            axes[j, i].axis('off')
    
    plt.suptitle("Preprocessing Steps for Perceptual Hashing Method", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def phash_method():
    train_hashes, train_labels, train_samples, reference_hashes, train_gray, train_equalized, train_thresh = load_dataset_phash(
        TRAIN_PATH, TRAIN_ANNOTATION, LOT_DATES)
    val_hashes, val_labels, _, _, _, _, _ = load_dataset_phash(
        VALID_PATH, VALID_ANNOTATION, LOT_DATES)
    test_hashes, test_labels, _, _, _, _, _ = load_dataset_phash(
        TEST_PATH, TEST_ANNOTATION, LOT_DATES)
    
    optimal_threshold = tune_threshold(val_hashes, val_labels, reference_hashes)
    print(f"Optimal Hamming Distance Threshold: {optimal_threshold}")
    
    plot_samples_phash(train_samples, train_gray, train_equalized, train_thresh)
    
    test_pred = [classify_phash(test_hash, reference_hashes, optimal_threshold) for test_hash in test_hashes]
    
    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred)
    
    print(f"Perceptual Hashing Method Accuracy: {accuracy:.4f}")
    print(f"Perceptual Hashing Method Precision: {precision:.4f}")
    print(f"Perceptual Hashing Method Recall: {recall:.4f}")
    print(f"Perceptual Hashing Method F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(test_labels, test_pred, "Perceptual Hashing Method")
    
    return accuracy





#--------------------------------------------Method 3------------------------------------------------------
#--------------------------------------------SVM------------------------------------------------------
# SVM Method
def preprocess_image_svm(image, bbox, target_size=IMG_SIZE):
    x, y, w, h = map(int, bbox)
    cropped = image[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, transform_sqrt=True)
    return fd, resized, gray, hog_image

def load_dataset_svm(data_path, annotation_path, date_range=None):
    image_map, annotations = load_coco_annotations(annotation_path)
    features, labels, sample_images, sample_gray_images, sample_hog_images = [], [], [], [], []
    empty_count, occupied_count = 0, 0
    
    for img_id, img_info in image_map.items():
        if date_range and img_info['date']:
            if not (date_range[0] <= img_info['date'] <= date_range[1]):
                continue
        img_path = os.path.join(data_path, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        for ann in annotations.get(img_id, []):
            fd, resized_img, gray_img, hog_img = preprocess_image_svm(image, ann['bbox'])
            features.append(fd)
            labels.append(ann['label'])
            if ann['label'] == 0 and empty_count < 1:
                sample_images.append((resized_img, ann['label']))
                sample_gray_images.append(gray_img)
                sample_hog_images.append(hog_img)
                empty_count += 1
            elif ann['label'] == 1 and occupied_count < 1:
                sample_images.append((resized_img, ann['label']))
                sample_gray_images.append(gray_img)
                sample_hog_images.append(hog_img)
                occupied_count += 1
    
    return np.array(features), np.array(labels), sample_images, sample_gray_images, sample_hog_images

def plot_samples_svm(sample_images, sample_gray_images, sample_hog_images):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    titles = ['Original', 'Grayscale', 'HOG Features']
    
    for i, ((img, label), gray_img, hog_img) in enumerate(zip(sample_images, sample_gray_images, sample_hog_images)):
        images = [img, gray_img, hog_img]
        for j, (image, title) in enumerate(zip(images, titles)):
            if j == 0:
                axes[j, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[j, i].imshow(image, cmap='gray')
            axes[j, i].set_title(f"{title} {'Occupied' if label == 1 else 'Empty'}")
            axes[j, i].axis('off')
    
    plt.suptitle("Preprocessing Steps for SVM Method", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def svm_method():
    train_features, train_labels, train_samples, train_gray, train_hog = load_dataset_svm(
        TRAIN_PATH, TRAIN_ANNOTATION, LOT_DATES)
    test_features, test_labels, _, _, _ = load_dataset_svm(
        TEST_PATH, TEST_ANNOTATION, LOT_DATES)
    
    plot_samples_svm(train_samples, train_gray, train_hog)
    
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(train_features, train_labels)
    
    test_pred = svm.predict(test_features)
    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred)
    
    print(f"SVM Method Accuracy: {accuracy:.4f}")
    print(f"SVM Method Precision: {precision:.4f}")
    print(f"SVM Method Recall: {recall:.4f}")
    print(f"SVM Method F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(test_labels, test_pred, "SVM Method")
    
    return accuracy






#--------------------------------------------Method 4------------------------------------------------------
#--------------------------------------------mAlexNet------------------------------------------------------
# mAlexNet Method
def preprocess_image_malexnet(image, bbox, target_size=(64, 64)):
    x, y, w, h = map(int, bbox)
    cropped = image[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    return rgb, resized

def load_dataset_malexnet(data_path, annotation_path, date_range=None):
    image_map, annotations = load_coco_annotations(annotation_path)
    images, labels, sample_images = [], [], []
    empty_count, occupied_count = 0, 0
    
    for img_id, img_info in image_map.items():
        if date_range and img_info['date']:
            if not (date_range[0] <= img_info['date'] <= date_range[1]):
                continue
        img_path = os.path.join(data_path, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        for ann in annotations.get(img_id, []):
            rgb_img, resized_img = preprocess_image_malexnet(image, ann['bbox'])
            images.append(rgb_img)
            labels.append(ann['label'])
            if ann['label'] == 0 and empty_count < 1:
                sample_images.append((resized_img, ann['label']))
                empty_count += 1
            elif ann['label'] == 1 and occupied_count < 1:
                sample_images.append((resized_img, ann['label']))
                occupied_count += 1
    
    return np.array(images), np.array(labels), sample_images

def build_malexnet():
    model = Sequential([
        Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNormalization(),
        
        Conv2D(96, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        
        Conv2D(96, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def plot_samples_malexnet(sample_images):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    titles = ['Original (Cropped, Resized)', 'RGB Normalized']
    
    for i, (img, label) in enumerate(sample_images):
        rgb_img, _ = preprocess_image_malexnet(img, [0, 0, img.shape[1], img.shape[0]])
        images = [img, rgb_img]
        for j, (image, title) in enumerate(zip(images, titles)):
            axes[j, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[j, i].set_title(f"{title} {'Occupied' if label == 1 else 'Empty'}")
            axes[j, i].axis('off')
    
    plt.suptitle(f"Preprocessing Steps for mAlexNet Method\n(Image Size: {IMG_SIZE}, Normalized to [0,1])", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def malexnet_method():
    train_images, train_labels, train_samples = load_dataset_malexnet(
        TRAIN_PATH, TRAIN_ANNOTATION, LOT_DATES)
    valid_images, valid_labels, _ = load_dataset_malexnet(
        VALID_PATH, VALID_ANNOTATION, LOT_DATES)
    test_images, test_labels, _ = load_dataset_malexnet(
        TEST_PATH, TEST_ANNOTATION, LOT_DATES)
    
    plot_samples_malexnet(train_samples)
    
    model = build_malexnet()
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(valid_images, valid_labels),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    test_pred = (model.predict(test_images) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred)
    
    print(f"mAlexNet Method Accuracy: {accuracy:.4f}")
    print(f"mAlexNet Method Precision: {precision:.4f}")
    print(f"mAlexNet Method Recall: {recall:.4f}")
    print(f"mAlexNet Method F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(test_labels, test_pred, "mAlexNet Method")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (mAlexNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (mAlexNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return accuracy







#------------------------------------------Performance Metrix------------------------------------------------------
# Shared confusion matrix plotting
def plot_confusion_matrix(y_true, y_pred, method_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"{method_name} Confusion Matrix:")
    print("[[True Negatives (Empty)  False Positives (Empty as Occupied)]")
    print(" [False Negatives (Occupied as Empty)  True Positives (Occupied)]]")
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Empty', 'Occupied'], 
                yticklabels=['Empty', 'Occupied'])
    plt.title(f'Confusion Matrix - {method_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return cm

# Compare accuracies across methods
def plot_accuracy_comparison(accuracies):
    methods = ['Traditional', 'Perceptual Hashing', 'SVM', 'mAlexNet']
    plt.figure(figsize=(8, 6))
    plt.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Accuracy Comparison Across Methods', fontsize=16)
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()








#--------------------------------------------Main------------------------------------------------------
# Main function to select method
def main(method='all'):
    accuracies = []
    if method.lower() == 'traditional' or method.lower() == 'all':
        print("Running Traditional (Method 1)")
        accuracies.append(traditional_method())
    if method.lower() == 'phash' or method.lower() == 'all':
        print("Running Perceptual Hashing (Method 2)")
        accuracies.append(phash_method())
    if method.lower() == 'svm' or method.lower() == 'all':
        print("Running SVM (Method 3)")
        accuracies.append(svm_method())
    if method.lower() == 'malexnet' or method.lower() == 'all':
        print("Running mAlexNet (Method 4).")
        accuracies.append(malexnet_method())
    
    if method.lower() == 'all':
        plot_accuracy_comparison(accuracies)

if __name__ == "__main__":
    main('all')