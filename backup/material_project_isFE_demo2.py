from mp_api.client import MPRester
import os

#API 密钥
API_KEY = "4PWVdBf8wfrNNrJ4rV5aIgdHz4B2va5t"

# 定义介电常数阈值
e_total_threshold = 10  # 可根据需要调整阈值

with MPRester(API_KEY) as mpr:
    # 获取 dielectric 模块中的数据
    dielectric_results = mpr.materials.dielectric.search(
        fields=["material_id", "formula_pretty", "e_total", "symmetry.is_centrosymmetric", "spacegroup.symbol"]
    )

    # 筛选满足介电常数和非中心对称条件的材料
    ferroelectric_candidates = [
        material for material in dielectric_results
        if getattr(material, 'e_total', None) and material.e_total > e_total_threshold
        and not getattr(material.symmetry, 'is_centrosymmetric', True)  # 确保材料是非中心对称的
    ]

    # 打印结果
    print("Potential Ferroelectric Materials (符合铁电材料筛选条件):")
    for material in ferroelectric_candidates:
        e_total = getattr(material, 'e_total', "N/A")
        space_group = getattr(material, 'spacegroup.symbol', "N/A")
        is_centrosymmetric = getattr(material.symmetry, 'is_centrosymmetric', "N/A")
        print(f"Material ID: {material.material_id}")
        print(f"Formula: {material.formula_pretty}")
        print(f"Total Dielectric Constant (e_total): {e_total}")
        print(f"Space Group: {space_group}")
        print(f"Is Centrosymmetric: {is_centrosymmetric}")
        print("------------------------------------------------")



