import numpy as np
import cv2
import os 
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
def visualize(root,name,pred_mask):
    pred_mask = np.where(pred_mask>0.5,255,0)
    pred_mask = pred_mask.squeeze()
    # import ipdb
    # ipdb.set_trace()
    path = os.path.join(root,name)
    img_name = f'{path}.png'
    cv2.imwrite(img_name,pred_mask)
    print(f"{img_name} saved!")