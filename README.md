# Zinc Detection Using Deep Learning

This repository contains the complete source code used for the master's thesis titled:
**"Zinc Detection in Recyclable Metal Waste Using Deep Learning"**  for Lucerne University of Applied Sciences and Arts, conducted in collaboration with **ZAV Recycling AG**, Switzerland.
**Supervised by Prof. Dr. Christian Schucan and PRof. Dr. Peter Seifter.**

## Abstract
This thesis introduces an artificial intelligence (AI) based approach in automating the detection and sorting of zinc and its alloys from other metals during recycling process. Traditional sorting techniques like manual sorting are often labour-intensive and inefficient for accurately identifying specific materials. While magnetic separation is a great option for separating ferrous and non-ferrous metals, it is inadequate for separating specific metals like zinc. To address this gap, a custom image dataset was developed and used to train and evaluate five deep learning object detection models: YOLOv8, YOLOv5, Faster R-CNN, RetinaNet, and EfficientDet.

The models were evaluated using the standard evaluation metrics such as mAP@0.5, precision, recall, F1-Score, inference time, and model size. Evaluation plots such as precision-recall curves, IoU histograms, and confusion matrixes were also visualised. The best performing model among all the trained models was YOLOv8, which offered a high accuracy and speed with low resource requirements. In contrast, EfficientDet failed to learn from the data and did not yield valid results. While Faster R-CNN and Retina Net demonstrated high accuracy, they were limited by their larger size and processing requirements. 

This thesis also demonstrates the feasibility of using deep learning object detection for detecting specific metals like zinc and contributes a tailored and labelled zinc dataset. Future work will focus on deploying real-time sorting systems and expanding to multi-class detection.
