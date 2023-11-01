import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
import torch

def acc_score(output, target, batch_size):
    """
    计算分类准确度
    Args:
        output (Tensor): 模型的输出，形状为 (batch_size, num_classes)
        target (Tensor): 目标标签，形状为 (batch_size,)
        batch_size (int): 当前批次的大小
    Returns:
        accuracy (float): 分类准确度
    """
    with torch.no_grad():
        # 获取预测类别
        output_probabilities = torch.exp(output)
        print("output_probabilities{}".format(output_probabilities))
        predicted = output_probabilities.argmax(dim=1)
        print("predicted{}".format(predicted))
        rel = target.argmax(dim=1)
        print("rel{}".format(rel))
        # 计算正确预测的数量
        correct = (predicted == rel).sum().item()
        print("correct{}".format(correct)) 
        # 计算准确度
        total = batch_size
        accuracy = correct / total

    return accuracy


