from mp_api.client import MPRester

# 替换为你的 API 密钥
API_KEY = "4PWVdBf8wfrNNrJ4rV5aIgdHz4B2va5t"

# 初始化 MPRester
with MPRester(API_KEY) as mpr:
    # 获取铁磁性材料数据
    ferromagnetic_results = mpr.materials.magnetism.search(
        fields=["material_id", "formula_pretty", "ordering", "total_magnetization", "curie_temperature"]
    )

    # 筛选出铁磁性材料（磁性排序为 FM）
    filtered_ferromagnetic_results = [
        material for material in ferromagnetic_results
        if material.ordering == "FM" and material.total_magnetization is not None # 磁化强度不为空
    ]

    print("Ferromagnetic Materials:")
    for material in filtered_ferromagnetic_results:
        curie_temp = getattr(material, "curie_temperature", None)
        print(f"Material ID: {material.material_id}")
        print(f"Formula: {material.formula_pretty}")
        print(f"Magnetic Ordering: {material.ordering}")
        print(f"Total Magnetization: {material.total_magnetization}")
        print(f"Curie Temperature: {curie_temp}")
        print("------------------------------------------------")


"""



from mp_api.client import MPRester

# 替换为你的 API 密钥
API_KEY = "4PWVdBf8wfrNNrJ4rV5aIgdHz4B2va5t"

# 初始化 MPRester
with MPRester(API_KEY) as mpr:
    # 获取铁磁性材料数据
    ferromagnetic_results = mpr.materials.magnetism.search(
        fields=["material_id", "formula_pretty", "ordering", "total_magnetization", "curie_temperature"]
    )

    # 筛选出铁磁性材料（磁性排序为 FM）和总磁化较高的材料
    filtered_ferromagnetic_results = [
        material for material in ferromagnetic_results
        if material.ordering == "FM" and material.total_magnetization is not None and material.total_magnetization > 0.1  # 磁化强度阈值
    ]

    print("Ferromagnetic Materials:")
    for material in filtered_ferromagnetic_results:
        curie_temp = getattr(material, "curie_temperature", None)
        print(f"Material ID: {material.material_id}")
        print(f"Formula: {material.formula_pretty}")
        print(f"Magnetic Ordering: {material.ordering}")
        print(f"Total Magnetization: {material.total_magnetization}")
        print(f"Curie Temperature: {curie_temp}")
        print("------------------------------------------------")
        
添加磁化强度阈值，筛选出总磁化大于 0.1 的铁磁性材料
"""
