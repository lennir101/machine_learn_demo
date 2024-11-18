from mp_api.client import MPRester
import pandas as pd
import numpy as np
from datetime import datetime
import os
from api_config import MP_API_KEY

def setup_output_directory(dir_name="material_search_results"):
    """创建输出目录"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def get_search_params():
    """获取搜索参数"""
    return {
        "magnetic_ordering": "FM",
        "min_magnetization": 0.1,
        "search_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def fetch_materials_data(mpr):
    """从Materials Project获取材料数据"""
    return mpr.materials.magnetism.search(
        fields=["material_id", "formula_pretty", "ordering", "total_magnetization", 
               "curie_temperature", "band_gap", "formation_energy_per_atom"]
    )

def filter_materials(materials, min_magnetization=0.1):
    """筛选符合条件的材料并处理缺失值"""
    filtered_materials = []
    for material in materials:
        if (material.ordering == "FM" and 
            material.total_magnetization is not None and 
            material.total_magnetization > min_magnetization):
            
            material_data = {
                "Material ID": material.material_id or "N/A",
                "Formula": material.formula_pretty or "N/A",
                "Magnetic Ordering": material.ordering or "N/A",
                "Total Magnetization (μB)": material.total_magnetization or "N/A",
                "Curie Temperature (K)": getattr(material, "curie_temperature", "N/A"),
                "Band Gap (eV)": getattr(material, "band_gap", "N/A"),
                "Formation Energy (eV/atom)": getattr(material, "formation_energy_per_atom", "N/A")
            }
            filtered_materials.append(material_data)
    return filtered_materials

def save_results(filtered_materials, search_params, output_dir):
    """保存结果到CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(filtered_materials)
    
    # 替换DataFrame中的None和NaN值为"N/A"
    df = df.replace({None: "N/A", np.nan: "N/A"})
    
    # 生成文件名
    filename = f"ferromagnetic_materials_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 保存数据
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    return filepath, df

def print_statistics(df, filepath):
    """打印详细的统计信息"""
    print(f"\n搜索完成！共找到 {len(df)} 个符合条件的材料")
    print(f"结果已保存至：{filepath}")
    
    print("\n=== 数据统计 ===")
    
    # 磁化强度统计
    mag_values = pd.to_numeric(df['Total Magnetization (μB)'].replace('N/A', np.nan), errors='coerce')
    print("\n磁化强度统计：")
    print(f"平均值：{mag_values.mean():.2f} μB")
    print(f"最大值：{mag_values.max():.2f} μB")
    print(f"最小值：{mag_values.min():.2f} μB")
    print(f"中位数：{mag_values.median():.2f} μB")
    
    # 居里温度统计
    curie_values = pd.to_numeric(df['Curie Temperature (K)'].replace('N/A', np.nan), errors='coerce')
    if not curie_values.isna().all():
        print("\n居里温度统计：")
        print(f"平均值：{curie_values.mean():.2f} K")
        print(f"最大值：{curie_values.max():.2f} K")
        print(f"最小值：{curie_values.min():.2f} K")
        print(f"中位数：{curie_values.median():.2f} K")
        print(f"数据完整度：{(~curie_values.isna()).mean():.1%}")
    
    # 能带间隙统计
    gap_values = pd.to_numeric(df['Band Gap (eV)'].replace('N/A', np.nan), errors='coerce')
    if not gap_values.isna().all():
        print("\n能带间隙统计：")
        print(f"平均值：{gap_values.mean():.2f} eV")
        print(f"数据完整度：{(~gap_values.isna()).mean():.1%}")
    
    # 数据完整性统计
    print("\n数据完整性统计：")
    for column in df.columns:
        valid_ratio = (df[column] != 'N/A').mean()
        print(f"{column}: {valid_ratio:.1%}")
    
    # 显示最强磁化材料
    print("\n磁化强度最强的前5个材料：")
    top_materials = df.nlargest(5, 'Total Magnetization (μB)')[
        ['Formula', 'Total Magnetization (μB)', 'Curie Temperature (K)']
    ]
    print(top_materials.to_string(index=False))

def main():
    """主函数"""
    # 设置输出目录
    output_dir = setup_output_directory()
    
    # 获取搜索参数
    search_params = get_search_params()
    
    try:
        # 执行材料搜索和数据处理
        with MPRester(MP_API_KEY) as mpr:
            # 获取材料数据
            materials = fetch_materials_data(mpr)
            
            # 筛选材料
            filtered_materials = filter_materials(
                materials, 
                min_magnetization=search_params["min_magnetization"]
            )
            
            # 保存结果
            filepath, df = save_results(
                filtered_materials, 
                search_params, 
                output_dir
            )
            
            # 打印统计信息
            print_statistics(df, filepath)
            
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    main() 