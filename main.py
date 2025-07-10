from data_generator import prepare_data
from model import Autoencoder
from train import train_autoencoder
from detector import detect_anomalies, determine_threshold
from utils import evaluate_performance, visualize_errors



def main():
    # 准备数据
    train_loader, test_loader, X_test, y_test = prepare_data()

    # 初始化模型
    input_dim = 10  # 数据特征维度
    model = Autoencoder(input_dim)

    # 训练模型
    trained_model = train_autoencoder(model, train_loader)

    # 确定阈值
    threshold, _ = determine_threshold(trained_model, train_loader)
    print(f"设定的异常检测阈值: {threshold:.6f}")

    # 检测异常
    anomalies, errors = detect_anomalies(trained_model, test_loader, threshold)

    # 评估性能
    metrics = evaluate_performance(y_test, anomalies)

    # 可视化结果
    visualize_errors(errors, y_test, threshold)


if __name__ == "__main__":
    main()