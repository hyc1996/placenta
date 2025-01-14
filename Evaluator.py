import cv2
import numpy as np

def pixel_accuracy(y_true, y_pred):
    """计算像素准确率"""
    return np.sum(y_true == y_pred) / y_true.size

def calculate_iou(mask_true, mask_pred):
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou




