import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"准确率: {accuracy:.2%}")
    print(f"精确率: {precision:.2%}")
    print(f"召回率: {recall:.2%}")
    print(f"F1分数: {f1:.2f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def visualize_errors(errors, y_true, threshold):
    plt.figure(figsize=(10, 6))
    plt.hist(errors[y_true == 0], bins=50, label='正常数据', alpha=0.7)
    plt.hist(errors[y_true == 1], bins=50, label='异常数据', alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', label='阈值')
    plt.xlabel('重建误差 (MSE)')
    plt.ylabel('样本数量')
    plt.title('自编码器异常检测的重建误差分布')
    plt.legend()
    plt.show()