"""
铁磁性材料数据获取与分析工具
============================

功能概述
-------
本工具通过 Materials Project API 获取铁磁性材料的关键特性数据，并进行系统化的数据分析。
主要用于材料科学研究人员筛选和分析潜在的铁磁性材料。

数据获取与分析流程
---------------
1. 数据获取
   - 通过 Materials Project API 获取材料数据
   - 获取的关键特性包括：
     * material_id: 材料唯一标识
     * formula_pretty: 化学式
     * ordering: 磁性排序类型
     * total_magnetization: 总磁化强度 (μB)
     * curie_temperature: 居里温度 (K)
     * band_gap: 能带间隙 (eV)
     * 其他物理特性（密度、体积等）

2. 数据分析维度
   - 磁性特征分析
     * 总磁化强度分布
     * 居里温度分布
     * 磁性类型统计
   - 结构特征分析
     * 密度分布
     * 体积分布
   - 电子特征分析
     * 能带间隙分布
     * 形成能分布

数据解读指南
----------
1. 磁性材料筛选标准：
   - 总磁化强度 > 0：表示材料具有铁磁性
   - 居里温度越高越好：表示材料在更高温度下保持磁性
   - 形成能越低越好：表示材料越稳定

2. 重要参数解释：
   - total_magnetization（总磁化强度）
     * 单位：μB（玻尔磁子）
     * 含义：材料的净磁矩
     * 应用：评估材料的磁性强度
   
   - curie_temperature（居里温度）
     * 单位：K（开尔文）
     * 含义：材料失去铁磁性的温度
     * 应用：评估材料的实用温度范围
   
   - band_gap（能带间隙）
     * 单位：eV（电子伏特）
     * 含义：价带顶与导带底之间的能量差
     * 应用：评估材料的导电性

3. 数据分析结果说明：
   - 相关性分析：
     * 热图显示不同特性之间的相关程度
     * 相关系数范围：-1到1
     * 正值表示正相关，负值表示负相关
   
   - 分布分析：
     * 直方图显示各特性的数值分布
     * 可用于识别异常值和数据模式
   
   - 统计指标：
     * 均值：反映数据的集中趋势
     * 标准差：反映数据的离散程度
     * 偏度：反映数据分布的对称性
     * 峰度：反映数据分布的尖峭程度

应用场景
-------
1. 材料筛选：快速识别具有潜在应用价值的铁磁性材料
2. 数据挖掘：发现材料特性之间的关联规律
3. 机器学习：为材料性能预测提供训练数据
4. 研究参考：为材料科学研究提供数据支持

输出文件说明
----------
1. 原始数据（materials_data_raw_*.csv）
   - 包含所有未经处理的原始数据
   - 适用于深入分析和自定义处理

2. 处理后数据（materials_data_processed_*.csv）
   - 经过基础处理的数据集
   - 适用于直接进行机器学习建模

3. 分析结果（analysis_results/）
   - 包含各类统计图表和分析报告
   - 用于可视化理解数据特征
"""

from mp_api.client import MPRester
import pandas as pd
import numpy as np
from datetime import datetime
import os
from api_config import MP_API_KEY
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def setup_output_directory(dir_name="material_search_results"):
    """创建输出目录"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def get_available_fields(mpr):
    """
    获取Materials Project API可用的字段
    
    参数:
        mpr: Materials Project API客户端实例
    
    返回:
        list: 可用字段列表
    
    说明:
        - 获取API支持的所有材料属性字段
        - 打印字段列表供参考
        - 用于验证我们需要查询的字段是否可用
    """
    available_fields = mpr.materials.magnetism.available_fields
    print("可用的字段：")
    for field in available_fields:
        print(field)
    return available_fields

def get_valid_fields():
    """
    定义并验证所需的材料属性字段
    
    返回:
        list: 需要查询的字段列表
    
    字段说明:
        - material_id: 材料的唯一标识符
        - formula_pretty: 材料的化学式
        - ordering: 磁性排序类型（FM：铁磁性，AFM：反铁磁性等）
        - total_magnetization: 总磁化强度 (μB)
        - curie_temperature: 居里温度 (K)
        - band_gap: 能带间隙 (eV)
        - formation_energy_per_atom: 每原子形成能 (eV/atom)
        - density: 密度 (g/cm³)
        - volume: 体积 (Å³)
        - elasticity: 弹性属性
        - piezoelectric: 压电性能
        - dielectric: 介电性质
        - refractive_index: 折射率
        - conductivity: 电导率
        - is_magnetic: 是否具有磁性
        - magnetic_type: 磁性类型
    """
    return [
        "material_id", "formula_pretty", "ordering", "total_magnetization",
        "curie_temperature", "band_gap", "formation_energy_per_atom",
        "density", "volume", "elasticity", "piezoelectric", "dielectric",
        "refractive_index", "conductivity", "is_magnetic", "magnetic_type"
    ]

def fetch_materials_data(mpr, valid_fields):
    """从Materials Project获取材料数据"""
    return mpr.materials.magnetism.search(fields=valid_fields)

def process_materials_data(materials, valid_fields):
    """
    处理材料数据，统一处理缺失值
    
    参数:
        materials: API返回的原始材料数据
        valid_fields: 需要处理的有效字段列表
    
    返回:
        list: 处理后的材料数据列表
    
    处理说明:
        1. 遍历每个材料对象
        2. 提取指定字段的值
        3. 将None值替换为"N/A"
        4. 保持数据格式统一
    """
    processed_data = []
    for material in materials:
        material_data = {field: getattr(material, field, "N/A") for field in valid_fields}
        processed_data.append(material_data)
    return processed_data

def analyze_data(df):
    """
    详细的数据分析，为机器学习准备
    
    参数:
        df: pandas DataFrame，包含材料数据
    
    分析内容:
        1. 基础统计
           - 总样本数
           - 可用特征列表
        
        2. 类别特征分析
           - 统计每个类别的分布
           - 计算各类别的比例
        
        3. 数值特征分析
           - 描述性统计（均值、中位数、标准差等）
           - 相关性分析（热图）
           - 分布分析（直方图）
           - 计算偏度和峰度
        
        4. 数据完整性分析
           - 缺失值比例
           - 唯一值数量
           - 数据有效性评估
        
        5. 特征工程建议
           - 类别特征编码建议
           - 数值特征转换建议
           - 标准化/归一化建议
    
    输出:
        - 统计结果打印到控制台
        - 图表保存到analysis_results目录
        - 统计数据保存为CSV文件
    """
    output_dir = "analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n=== 数据分析结果 ===")
    print(f"\n总材料数量: {len(df)}")
    
    # 打印可用的列
    print("\n可用的列：")
    print(df.columns.tolist())
    
    # 2. 类别特征分析
    categorical_columns = ['magnetic_type', 'ordering', 'is_magnetic']
    available_categorical = [col for col in categorical_columns if col in df.columns]
    
    for col in available_categorical:
        print(f"\n{col} 分布:")
        distribution = df[col].value_counts()
        print(distribution)
        print(f"{col} 分布比例:")
        print(distribution / len(df))
    
    # 3. 数值特征分析
    all_numeric_columns = ['total_magnetization', 'band_gap', 'density', 'volume',
                          'formation_energy_per_atom', 'curie_temperature']
    # 只选择实际存在的列
    numeric_columns = [col for col in all_numeric_columns if col in df.columns]
    
    if numeric_columns:
        numeric_data = df[numeric_columns].copy()
        
        # 将'N/A'转换为np.nan
        for col in numeric_columns:
            numeric_data[col] = pd.to_numeric(numeric_data[col].replace('N/A', np.nan), errors='coerce')
        
        # 3.1 描述性统计
        print("\n数值特征描述性统计:")
        stats_df = numeric_data.describe()
        print(stats_df)
        stats_df.to_csv(os.path.join(output_dir, f'numeric_stats_{timestamp}.csv'))
        
        # 3.2 相关性分析
        if len(numeric_columns) > 1:  # 只在有多个数值列时进行相关性分析
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('特征相关性热图')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{timestamp}.png'))
            plt.close()
        
        # 3.3 分布分析
        for col in numeric_columns:
            values = numeric_data[col].dropna()
            if len(values) > 0:
                print(f"\n{col} 统计:")
                print(f"平均值: {values.mean():.2f}")
                print(f"中位数: {values.median():.2f}")
                print(f"标准差: {values.std():.2f}")
                try:
                    print(f"偏度: {stats.skew(values):.2f}")
                    print(f"峰度: {stats.kurtosis(values):.2f}")
                except:
                    print("无法计算偏度和峰度")
                print(f"数据完整度: {(~values.isna()).mean():.1%}")
                
                # 绘制分布图
                plt.figure(figsize=(10, 6))
                sns.histplot(values, kde=True)
                plt.title(f'{col} 分布')
                plt.savefig(os.path.join(output_dir, f'{col}_distribution_{timestamp}.png'))
                plt.close()
    
    # 4. 数据完整性分析
    completeness = pd.DataFrame({
        'column': df.columns,
        'missing_ratio': df.isna().mean(),
        'na_ratio': (df == 'N/A').mean(),
        'unique_values': df.nunique()
    })
    print("\n数据完整性分析:")
    print(completeness)
    completeness.to_csv(os.path.join(output_dir, f'data_completeness_{timestamp}.csv'))
    
    # 5. 特征工程建议
    print("\n=== 机器学习特征工程建议 ===")
    print("1. 类别特征编码建议:")
    for col in available_categorical:
        unique_count = df[col].nunique()
        if unique_count <= 5:
            print(f"   - {col}: 建议使用One-Hot编码")
        else:
            print(f"   - {col}: 建议使用Label编码或Target编码")
    
    if numeric_columns:
        print("\n2. 数值特征处理建议:")
        for col in numeric_columns:
            values = numeric_data[col].dropna()
            if len(values) > 0:
                try:
                    skewness = stats.skew(values)
                    if abs(skewness) > 1:
                        print(f"   - {col}: 建议进行对数转换或Box-Cox转换")
                except:
                    pass
                if values.std() > 100:
                    print(f"   - {col}: 建议进行标准化")
                elif values.max() / values.min() > 10:
                    print(f"   - {col}: 建议进行归一化")

def save_results(data, output_dir):
    """
    保存原始和处理后的数据到CSV文件
    
    参数:
        data: 原始材料数据
        output_dir: 输出目录路径
    
    返回:
        tuple: (原始数据DataFrame, 处理后的DataFrame)
    
    处理步骤:
        1. 数据转换
           - 将原始数据转换为DataFrame
           - 生成时间戳用于文件命名
        
        2. 特征工程
           - 类别特征编码
           - 数值特征处理
           - 缺失值处理
        
        3. 文件保存
           - 保存原始数据
           - 保存处理后的数据
           - 使用时间戳区分不同批次的数据
    
    特征处理说明:
        类别特征:
            - magnetic_type: 磁性类型分类编码
            - ordering: 磁性排序类型编码
            - is_magnetic: 布尔值编码
        
        数值特征:
            - total_magnetization: 总磁化强度
            - band_gap: 能带间隙
            - density: 密度
            - volume: 体积
            - formation_energy_per_atom: 形成能
            - curie_temperature: 居里温度
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(data)
    
    # 生成原始数据文件名
    raw_filename = f"materials_data_raw_{timestamp}.csv"
    raw_filepath = os.path.join(output_dir, raw_filename)
    
    # 保存原始数据
    df.to_csv(raw_filepath, index=False, encoding='utf-8')
    print(f"\n原始数据已保存至: {raw_filepath}")
    
    # 生成处理后的数据文件名
    processed_filename = f"materials_data_processed_{timestamp}.csv"
    processed_filepath = os.path.join(output_dir, processed_filename)
    
    # 基本的特征工程
    processed_df = df.copy()
    
    # 处理类别特征
    categorical_columns = ['magnetic_type', 'ordering', 'is_magnetic']
    available_categorical = [col for col in categorical_columns if col in processed_df.columns]
    
    for col in available_categorical:
        if col in processed_df.columns:
            # 将类别特征转换为数值
            processed_df[col] = pd.Categorical(processed_df[col]).codes
    
    # 处理数值特征
    numeric_columns = ['total_magnetization', 'band_gap', 'density', 'volume',
                      'formation_energy_per_atom', 'curie_temperature']
    available_numeric = [col for col in numeric_columns if col in processed_df.columns]
    
    for col in available_numeric:
        processed_df[col] = pd.to_numeric(processed_df[col].replace('N/A', np.nan), errors='coerce')
    
    # 保存处理后的数据
    processed_df.to_csv(processed_filepath, index=False, encoding='utf-8')
    print(f"处理后的数据已保存至: {processed_filepath}")
    
    return df, processed_df

def main():
    """主函数"""
    try:
        # 设置输出目录
        output_dir = setup_output_directory()
        
        with MPRester(MP_API_KEY) as mpr:
            # 获取可用字段
            available_fields = get_available_fields(mpr)
            
            # 获取有效字段
            requested_fields = get_valid_fields()
            valid_fields = [field for field in requested_fields if field in available_fields]
            
            # 获取材料数据
            materials = fetch_materials_data(mpr, valid_fields)
            
            # 处理数据
            processed_data = process_materials_data(materials, valid_fields)
            
            # 保存结果
            df, processed_df = save_results(processed_data, output_dir)
            
            # 分析数据
            analyze_data(df)
            
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
