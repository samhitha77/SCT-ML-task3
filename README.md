## SCT-ML- TASK3
# 🐱🐶 Cat vs Dog Image Classification using Support Vector Machine (SVM)

## 📌 Project Overview
This project implements a **Support Vector Machine (SVM)** classifier to distinguish between images of **cats and dogs**. Images are preprocessed using **OpenCV**, converted into **Histogram of Oriented Gradients (HOG)** features, and then classified using an SVM model built with **Scikit-learn**.

This project was completed as **Task 03** of the **SkillCraft Technology Machine Learning Internship**.

---

## 🎯 Objective
The objective of this project is to build a machine learning model capable of accurately classifying cat and dog images using image feature extraction and the Support Vector Machine algorithm.

---

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image (HOG)
- Matplotlib
- Joblib

---

## 📂 Dataset
**Microsoft Cats vs Dogs Dataset**

The dataset contains labeled images of cats and dogs and is available on **Kaggle**. link : https://www.kaggle.com/competitions/dogs-vs-cats/data

---

## ⚙️ Workflow

1. Load the dataset.
2. Resize all images to a fixed size.
3. Convert images to grayscale.
4. Extract HOG features.
5. Split the dataset into training and testing sets.
6. Train an SVM classifier.
7. Predict image labels.
8. Evaluate model performance using Accuracy, Confusion Matrix, and Classification Report.
9. Save the trained model using Joblib.

---

## 📊 Results

The trained SVM model successfully classifies cat and dog images using HOG feature extraction. Model performance is evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

---
---

## 📸 Output

The project generates:

- Sample Images
- Confusion Matrix
- Classification Report
- Trained SVM Model (`svm_model.pkl`)

---
---

## ✅ Conclusion

This project demonstrates how **Support Vector Machines (SVM)** can effectively classify images when combined with **Histogram of Oriented Gradients (HOG)** feature extraction. The model successfully learns distinguishing features between cats and dogs and achieves reliable classification performance. This project strengthens practical knowledge of image preprocessing, feature engineering, supervised machine learning, and model evaluation, providing a strong foundation for more advanced computer vision and deep learning applications.
