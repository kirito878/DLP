import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    epsilon = 1e-6
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    dice_score = (2.0*intersection+epsilon) / (union+epsilon)
    return dice_score


def mean_dice_score(predictions, ground_truth):
    total_dice = 0
    for pred_mask, gt_mask in zip(predictions, ground_truth):
        dice = dice_score(pred_mask, gt_mask)
        total_dice += dice

    dice_mean = total_dice / len(predictions)
    return dice_mean

# predict mask


def visualize(root, name, pred_mask):
    pred_mask = np.where(pred_mask > 0.5, 255, 0)
    pred_mask = pred_mask.squeeze()
    # import ipdb
    # ipdb.set_trace()
    path = os.path.join(root, name)
    img_name = f'{path}.jpg'
    cv2.imwrite(img_name, pred_mask)
    print(f"{img_name} saved!")


def visualize_overlap(save_root, img_root, name, pred_mask):
    rgb_path = os.path.join(img_root, name)
    rgb_path = f'{rgb_path}.jpg'
    pred_mask = np.where(pred_mask > 0.5, 255, 0)
    pred_mask = pred_mask.squeeze()
    h, w = pred_mask.shape
    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.resize(rgb_image, (h, w))
    mask_color = (255, 0, 0)
    pred_mask = cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pred_mask[np.where((pred_mask == [255, 255, 255]).all(axis=2))] = mask_color
    overlay = cv2.addWeighted(rgb_image, 0.5, pred_mask, 0.5, 0)
    path = os.path.join(save_root, name)
    img_name = f'{path}.jpg'
    cv2.imwrite(img_name, overlay)
    # print(f"{img_name} saved!")

def accuracy_score(pred_mask, gt_mask):
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = gt_mask.size
    accuracy = correct_pixels/total_pixels
    return accuracy


def mean_accuracy(predictions, ground_truth):
    total_accuracy = 0
    for pred_mask, gt_mask in zip(predictions, ground_truth):
        accuracy = accuracy_score(pred_mask, gt_mask)
        total_accuracy += accuracy
    accuracy_mean = total_accuracy / len(predictions) * 100
    return accuracy_mean


def show_result(epochs, unet_train_acc, unet_valid_acc, resnet34_unet_train_acc, resnet34_unet_valid_acc):
    plt.plot(epochs, unet_train_acc, linestyle='-', label='unet_train_acc')
    plt.plot(epochs, unet_valid_acc, linestyle='-', label='unet_valid_acc')
    plt.plot(epochs, resnet34_unet_train_acc,
             linestyle='-', label='resnet34_unet_train_acc')
    plt.plot(epochs, resnet34_unet_valid_acc,
             linestyle='-', label='resnet34_unet_valid_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig('acc_curve.png')
