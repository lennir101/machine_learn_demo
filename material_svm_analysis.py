"""
材料性质机器学习预测分析工具
============================

本程序用于分析和预测材料的磁性质，使用支持向量机(SVM)和随机森林(Random Forest)进行建模。

主要功能：
---------
1. 数据加载与预处理
2. 特征工程与选择
3. 多模型训练与对比
4. 模型评估与可视化
5. 预测结果分析
6. 交叉验证与性能评估

技术实现思路：
------------
1. 数据处理流程：
   - 加载CSV格式的材料数据
   - 处理缺失值和异常值
   - 特征标准化
   - 数据集划分（训练集/测试集）

2. 模型训练策略：
   - 实现多模型并行训练
   - 使用GridSearchCV进行超参数优化
   - 支持SVR和RandomForest两种模型
   - 实时显示训练进度
   
3. 评估指标体系：
   - R²得分（决定系数）
   - MSE（均方误差）
   - RMSE（均方根误差）
   - MAE（平均绝对误差）
   - 可解释方差得分

4. 可视化分析：
   - 学习曲线分析
   - 模型性能对比图
   - 预测值vs实际值散点图
   - 残差分析图
   - 特征重要性分析（随机森林）
   - 预测置信区间可视化

使用方法：
--------
1. 数据准备：
   - 准备包含材料特征和目标变量的CSV文件
   - 确保数据中包含必要的特征列（density, volume等）
   - 确保数据中包含目标变量列（total_magnetization）

2. 环境配置：
   需要安装以下Python包：
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - tqdm
   - joblib

3. 运行方式：
   直接运行python material_svm_analysis.py

输出说明：
--------
1. 模型文件：
   - 保存最佳模型（.joblib格式）

2. 可视化结果：
   - 学习曲线图
   - 模型对比图
   - 预测散点图
   - 残差分析图
   - 特征重要性图

3. 评估报告：
   - 详细的模型评估指标
   - 交叉验证结果
   - 各模型的最佳参数

代码结构：
--------
1. 数据处理模块：
   - load_and_prepare_data(): 数据加载和预处理
   - prepare_features(): 特征准备和标准化

2. 模型训练模块：
   - train_multiple_models(): 多模型训练和评估
   - perform_cross_validation(): 交叉验证

3. 可视化模块：
   - plot_learning_curves(): 学习曲线绘制
   - plot_model_comparison(): 模型对比图
   - plot_results(): 预测结果可视化
   - plot_residuals(): 残差分析
   - plot_feature_importance(): 特征重要性分析
   - plot_prediction_intervals(): 预测区间可视化

4. 评估模块：
   - evaluate_model_performance(): 模型性能评估

注意事项：
--------
1. 数据质量：
   - 确保输入数据的完整性
   - 处理异常值和缺失值
   - 特征的标准化处理

2. 计算资源：
   - GridSearchCV过程可能较为耗时
   - 建议使用多核CPU进行训练
   - 大数据集可能需要较大内存

3. 模型选择：
   - SVR适合非线性关系
   - RandomForest提供特征重要性分析
   - 可根据具体需求选择合适的模型

4. 结果解释：
   - 注意过拟合/欠拟合问题
   - 结合领域知识解释预测结果
   - 考虑预测的置信区间

维护信息：
--------
- 作者：[Jenson.Loh]
- 版本：1.0.0
- 最后更新：2024-01-17
- 许可证：MIT

"""

# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tqdm import tqdm
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
import joblib
from sklearn.metrics import mean_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid

# 在文件开头添加目录创建函数
def ensure_dir_exists(dir_path):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        dir_path: 目录路径
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")
        except Exception as e:
            raise Exception(f"创建目录失败 {dir_path}: {str(e)}")

def load_and_prepare_data(filepath):
    """
    加载和预处理数据
    
    参数:
        filepath: 处理后的CSV文件路径
    """
    try:
        # 确保数据目录存在
        data_dir = os.path.dirname(filepath)
        if data_dir:
            ensure_dir_exists(data_dir)
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
            
        # 加载数据
        df = pd.read_csv(filepath)
        
        # 检查数据是否为空
        if df.empty:
            raise ValueError("加载的数据集为空")
        
        # 移除包含NaN的行
        df = df.dropna()
        
        # 检查处理后是否还有数据
        if df.empty:
            raise ValueError("删除缺失值后数据集为空")
        
        # 打印数据基本信息
        print("\n数据集信息:")
        print(f"样本数量: {len(df)}")
        print(f"特征数量: {len(df.columns)}")
        print("\n可用特征:")
        print(df.columns.tolist())
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("无法加载数据文件，文件可能为空")
    except Exception as e:
        raise Exception(f"加载数据时出错: {str(e)}")

def prepare_features(df, target_col, feature_cols=None):
    """
    准备特征和目标变量
    
    参数:
        df: 数据DataFrame
        target_col: 目标变量列名
        feature_cols: 要使用的特征列名列表（如果为None，使用所有数值列）
    
    返回:
        X: 特征矩阵
        y: 目标变量
        feature_names: 使用的特征名列表
    """
    # 如果未指定特征列，使用所有数值列
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 移除目标变量（如果在特征列中）
        if target_col in feature_cols:
            feature_cols.remove(target_col)
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols

def train_multiple_models(X, y):
    """
    训练和对比多个模型
    
    参数:
        X: 特征矩阵
        y: 目标变量
    
    返回:
        最佳模型和评估结果
    """
    models = {
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=42)
    }
    
    param_grids = {
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    }
    
    results = {}
    best_score = float('-inf')
    best_model = None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 计算总迭代次数
    total_iterations = sum(len(list(ParameterGrid(param_grid))) for param_grid in param_grids.values())
    
    # 创建总进度条
    with tqdm(total=total_iterations, desc="总体训练进度", position=0) as pbar:
        for model_name, model in models.items():
            print(f"\n{'-'*50}")
            print(f"开始训练 {model_name} 模型")
            
            # 获取当前模型的参数组合数
            param_combinations = list(ParameterGrid(param_grids[model_name]))
            
            # 创建子进度条显示当前模型的训练进度
            with tqdm(total=len(param_combinations), 
                     desc=f"{model_name}训练进度", 
                     position=1, 
                     leave=False) as sub_pbar:
                
                start_time = datetime.now()
                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                
                # 自定义回调函数来更新进度条
                def update_progress(param_idx):
                    sub_pbar.update(1)
                    pbar.update(1)
                
                # 训练模型
                grid_search.fit(X_train, y_train)
                training_time = datetime.now() - start_time
                
                print(f"\n{model_name} 训练完成:")
                print(f"训练时间: {training_time}")
                print(f"最佳参数: {grid_search.best_params_}")
                print(f"最佳交叉验证得分: {grid_search.best_score_:.3f}")
                
                y_pred = grid_search.predict(X_test)
                
                results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'model': grid_search.best_estimator_
                }
                
                if results[model_name]['r2_score'] > best_score:
                    best_score = results[model_name]['r2_score']
                    best_model = grid_search.best_estimator_
    
    return best_model, results, (X_test, y_test, y_pred)

def plot_learning_curves(model, X, y, output_dir):
    """
    绘制学习曲线
    """
    # 确保输出目录存在
    ensure_dir_exists(output_dir)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=make_scorer(r2_score)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='训练集得分')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='验证集得分')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('训练样本数')
    plt.ylabel('R² 得分')
    plt.title('学习曲线')
    plt.legend(loc='best')
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'learning_curve_{timestamp}.png'))
    plt.close()

def plot_model_comparison(results, output_dir):
    """
    绘制模型对比图
    """
    # 确保输出目录存在
    ensure_dir_exists(output_dir)
    
    model_names = list(results.keys())
    r2_scores = [results[model]['r2_score'] for model in model_names]
    mse_scores = [results[model]['mse'] for model in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # R² 得分对比
    ax1.bar(model_names, r2_scores)
    ax1.set_title('模型 R² 得分对比')
    ax1.set_ylabel('R² 得分')
    
    # MSE 得分对比
    ax2.bar(model_names, mse_scores)
    ax2.set_title('模型 MSE 对比')
    ax2.set_ylabel('均方误差')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'model_comparison_{timestamp}.png'))
    plt.close()

def plot_results(results, target_col, output_dir):
    """
    绘制预测结果
    """
    # 确保输出目录存在
    ensure_dir_exists(output_dir)
    
    X_test, y_test, y_pred = results
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 预测vs实际值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{target_col} 预测vs实')
    plt.savefig(os.path.join(output_dir, f'prediction_scatter_{timestamp}.png'))
    plt.close()

def evaluate_model_performance(y_true, y_pred):
    """
    计算多个评估指标
    """
    metrics = {
        'R² 得': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        '可解释方差': explained_variance_score(y_true, y_pred)
    }
    return metrics

def plot_residuals(y_true, y_pred, output_dir):
    """
    绘制残差分析图
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 4))
    
    # 残差分布图
    plt.subplot(121)
    sns.histplot(residuals, kde=True)
    plt.title('残差分布')
    plt.xlabel('残差')
    
    # 残差vs预值
    plt.subplot(122)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差 vs 预测值')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'residuals_analysis_{timestamp}.png'))
    plt.close()

def plot_feature_importance(model, feature_names, output_dir):
    """
    绘制特征重要（适用于RandomForest模型）
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('特征重要性')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'feature_importance_{timestamp}.png'))
        plt.close()

def perform_cross_validation(model, X, y, cv=5):
    """
    执行交叉验证并返回详细结果
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_results = {
        '平均 R²': cv_scores.mean(),
        'R² 标准差': cv_scores.std(),
        '最大 R²': cv_scores.max(),
        '最小 R²': cv_scores.min(),
        '各折得分': cv_scores.tolist()
    }
    return cv_results

def plot_prediction_intervals(model, X_test, y_test, output_dir):
    """
    使用Bootstrap方法估计预测的置信区间（适用于RandomForest）
    """
    if hasattr(model, 'estimators_'):
        predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, mean_pred, alpha=0.5)
        plt.fill_between(y_test, 
                        mean_pred - 1.96 * std_pred,
                        mean_pred + 1.96 * std_pred,
                        alpha=0.2)
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('预测值及95%置信区间')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'prediction_intervals_{timestamp}.png'))
        plt.close()

def main():
    try:
        # 设置基本目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 确保所有必要的目录存在
        output_dir = os.path.join(base_dir, "svm_analysis_results")
        data_dir = os.path.join(base_dir, "material_search_results")
        ensure_dir_exists(output_dir)
        ensure_dir_exists(data_dir)
        
        print("开始数据分析...")
        data_path = os.path.join(data_dir, "materials_data_processed_20241117_110126.csv")
        
        # 如果数据文件不存在，给出友好提示
        if not os.path.exists(data_path):
            print(f"警告: 数据文件不存在: {data_path}")
            print("请确保数据文件位于正确位置，或修改文件路径")
            return
            
        df = load_and_prepare_data(data_path)
        
        target_col = 'total_magnetization'
        feature_cols = ['density', 'volume']
        
        # 检查特征列是否存在
        missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少以下列: {missing_cols}")
            
        print("准备特征数据...")
        X, y, feature_names = prepare_features(df, target_col, feature_cols)
        
        print(f"\n使用的特征: {feature_names}")
        print(f"目标变量: {target_col}")
        
        print("\n开始模型训练和评估...")
        best_model, all_results, (X_test, y_test, y_pred) = train_multiple_models(X, y)
        
        print("\n绘制评估图表...")
        plot_results((X_test, y_test, y_pred), target_col, output_dir)
        plot_learning_curves(best_model, X, y, output_dir)
        plot_model_comparison(all_results, output_dir)
        plot_residuals(y_test, y_pred, output_dir)
        plot_feature_importance(best_model, feature_names, output_dir)
        
        # 保存最佳模型
        model_path = os.path.join(output_dir, 'best_model.joblib')
        joblib.dump(best_model, model_path)
        
        print(f"\n分析结果已保存至: {output_dir}")
        print(f"最佳模型已保存至: {model_path}")
        
        # 打印所有模型的详细结果
        print("\n模型评估结果:")
        for model_name, result in all_results.items():
            print(f"\n{model_name}:")
            print(f"最佳参数: {result['best_params']}")
            print(f"R² 得分: {result['r2_score']:.3f}")
            print(f"均方误差: {result['mse']:.3f}")
        
        print("\n执行交叉验证...")
        cv_results = perform_cross_validation(best_model, X, y)
        print("\n交叉验证结果:")
        for metric, value in cv_results.items():
            print(f"{metric}: {value}")
        
        print("\n计算详细评估指标...")
        metrics = evaluate_model_performance(y_test, y_pred)
        print("\n详细评估指标:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
        
        print("\n绘制额外分析图表...")
        plot_residuals(y_test, y_pred, output_dir)
        if isinstance(best_model, RandomForestRegressor):
            plot_feature_importance(best_model, feature_names, output_dir)
            plot_prediction_intervals(best_model, X_test, y_test, output_dir)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 