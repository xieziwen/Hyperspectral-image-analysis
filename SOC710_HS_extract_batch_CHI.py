"""
Created on Wed Oct 22 09:11:41 2025

@author: Ziwen_Xie
"""
# --------------------------
# 1. 导入依赖库
# --------------------------
import numpy as np
import matplotlib.pyplot as plt 
import warnings
from spectral.io import envi
import re
from sklearn.cluster import DBSCAN
from skimage.morphology import closing, rectangle, remove_small_holes, remove_small_objects
import pandas as pd
from skimage.measure import label, regionprops
import os
import time

# --------------------------
# 2. 全局配置
# --------------------------
class Config:
    data_dir = "data"           
    result_dir = "result"       
    target_wls = {"G": 550.0, "R": 660.0, "RE": 750.0}
    threshold = 0.10  
    dbscan_eps = 12    
    dbscan_min_samples = 6  
    closing_struct_size = (9, 11)  
    hole_threshold = 400  
    min_object_size = 30  
    target_rgb_wavelengths = {"R": 700.0, "G": 546.1,  "B": 435.8}
    target_foreground_count = 0  # 目标叶片数量，0表示自动判断，>0表示指定数量
    separate_folder = False  # 是否将每个图像结果放在独立文件夹中
    
    # 新增：控制各步骤结果是否输出的参数（True=输出，False=不输出）
    output_settings = {
        "single_band": False,        # 单波段图像
        "rgb_composite": True,      # 三波段合成图
        "mcari_index": False,        # MCARI指数图
        "nmcari_index": False,       # 归一化MCARI指数图
        "threshold_mask": False,     # 阈值分割结果
        "denoised_mask": False,      # 聚类去噪结果
        "closed_mask": False,        # 闭运算结果
        "filled_mask": False,        # 补全空洞结果
        "final_mask": False,         # 最终目标叶片区域
        "process_comparison": True, # 全流程对比图
        "mask_overlay": False,       # 目标叶片区域掩膜叠加
        "foreground_only": True,    # 仅保留目标叶片区域
        "reflectance_data": True,   # 反射率数据CSV
        "target_labeling": True     # 目标叶片标记图
    }

# 过滤 matplotlib 字体找不到的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
# 新增：过滤字体查找相关的特定警告
warnings.filterwarnings("ignore", message="findfont: Font family.*not found.")

# 设置系统更可能存在的中文字体（优先使用 SimHei、Microsoft YaHei）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# --------------------------
# 3. 工具函数
# --------------------------
def find_closest_band(wavelengths, target_wl):
    if not wavelengths:
        return None
    diffs = np.abs(np.array(wavelengths) - target_wl)
    return np.argmin(diffs)


def extract_hdr_params(hdr_path):
    try:
        with open(hdr_path, 'r') as f:
            content = f.read().lower()
        lines = int(re.search(r'lines\s*=\s*(\d+)', content).group(1))
        samples = int(re.search(r'samples\s*=\s*(\d+)', content).group(1))
        bands = int(re.search(r'bands\s*=\s*(\d+)', content).group(1))
        return lines, samples, bands
    except Exception as e:
        print(f"[ERROR] 提取hdr参数失败：{str(e)}")
        return None, None, None


def extract_wavelengths(hdr_path):
    try:
        with open(hdr_path, 'r') as f:
            content = f.read()
        wave_match = re.search(r'wavelength\s*=\s*\{(.*?)\}', content, re.IGNORECASE | re.DOTALL)
        if not wave_match:
            print("[ERROR] hdr文件未找到wavelength字段")
            return None
        wave_str = wave_match.group(1).strip().replace('\n', '').replace(' ', '')
        wavelengths = list(map(float, wave_str.split(',')))
        return wavelengths
    except Exception as e:
        print(f"[ERROR] 提取波长失败：{str(e)}")
        return None


def plot_single_mask(mask, title, pixel_count, figsize=(10, 8), show_colorbar=True):
    plt.figure(figsize=figsize)
    plt.imshow(mask, cmap='gray')
    if show_colorbar:
        plt.colorbar(label='像素值（1=前景，0=背景）')
    plt.title(f'{title}（前景数={pixel_count}）', fontsize=12)
    plt.xlabel('列数', fontsize=10)
    plt.ylabel('行数', fontsize=10)
    plt.tight_layout()
    return plt


def plot_rgb_image(rgb_img, title, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(rgb_img)
    plt.title(title, fontsize=12)
    plt.xlabel(f'列数（{rgb_img.shape[1]}）', fontsize=10)
    plt.ylabel(f'行数（{rgb_img.shape[0]}）', fontsize=10)
    plt.tight_layout()
    return plt


def get_file_pairs(data_dir):
    if not os.path.exists(data_dir):
        print(f"[ERROR] 数据目录 {data_dir} 不存在")
        return []
    files = os.listdir(data_dir)
    float_files = [f for f in files if f.lower().endswith('.float')]
    base_names = [os.path.splitext(f)[0] for f in float_files]
    valid_pairs = []
    for base_name in base_names:
        hdr_file = os.path.join(data_dir, f"{base_name}.hdr")
        float_file = os.path.join(data_dir, f"{base_name}.float")
        if os.path.exists(hdr_file) and os.path.exists(float_file):
            valid_pairs.append((base_name, hdr_file, float_file))
        else:
            print(f"[WARN] 跳过不完整的文件对: {base_name}（缺少.hdr或.float文件）")
    return valid_pairs


def process_single_file(base_name, hdr_path, float_path, result_dir):
    print(f"\n[INFO] 开始处理文件: {base_name}")
    
    # 根据配置决定结果保存目录
    if Config.separate_folder:
        file_result_dir = os.path.join(result_dir, base_name)
    else:
        file_result_dir = result_dir
    os.makedirs(file_result_dir, exist_ok=True)
    
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # 读取hdr参数
    lines_hdr, samples_hdr, bands_hdr = extract_hdr_params(hdr_path)
    if None in [lines_hdr, samples_hdr, bands_hdr]:
        print(f"[ERROR] 无法从 {base_name}.hdr 提取完整维度信息，跳过该文件")
        return
    rows, cols, bands = lines_hdr, samples_hdr, bands_hdr
    print(f"[INFO] 图像参数：行数={rows}, 列数={cols}, 波段数={bands}")

    # 读取高光谱数据
    try:
        img = envi.open(hdr_path, float_path)
        data_cube = img.load()
        total_elements = rows * cols * bands
        if data_cube.size != total_elements:
            raise ValueError(f"数据尺寸不匹配（实际{data_cube.size}，预期{total_elements}）")
        data_cube = data_cube.reshape((rows, cols, bands))
        print(f"[INFO] 重塑后数据形状：{data_cube.shape}")
    except Exception as e:
        print(f"[ERROR] 数据读取失败：{str(e)}，跳过该文件")
        return

    # 单波段图像保存
    band_to_show = min(49, bands - 1)
    band_data = np.squeeze(data_cube[:, :, band_to_show])
    plt.figure(figsize=(10, 8))
    plt.imshow(band_data, cmap='gray', vmin=np.percentile(band_data, 1), vmax=np.percentile(band_data, 99))
    plt.colorbar(label='反射率值')
    plt.title(f'高光谱单波段图像（{cols}×{rows}）- 第{band_to_show+1}个波段', fontsize=12)
    plt.xlabel(f'列数（{cols}）', fontsize=10)
    plt.ylabel(f'行数（{rows}）', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["single_band"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_单波段图像.png"), dpi=300)
    plt.close()

    # 三波段合成图（目标波长匹配）
    rgb_img = None
    if bands >= 3:
        wavelengths = extract_wavelengths(hdr_path)
        if wavelengths is not None:
            rgb_band_indices = {}
            for channel, target_wl in Config.target_rgb_wavelengths.items():
                idx = find_closest_band(wavelengths, target_wl)
                if idx is not None:
                    idx = min(idx, bands - 1)
                    rgb_band_indices[channel] = idx
                    print(f"[INFO] {channel}通道：目标{target_wl}nm → 匹配{wavelengths[idx]}nm（索引{idx}）")
            if len(rgb_band_indices) == 3:
                r_band = rgb_band_indices["R"]
                g_band = rgb_band_indices["G"]
                b_band = rgb_band_indices["B"]
                def normalize(x):
                    x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
                    return (x - x_min) / (x_max - x_min) if x_max != x_min else x
                rgb_img = np.stack([
                    normalize(data_cube[:, :, r_band]), 
                    normalize(data_cube[:, :, g_band]), 
                    normalize(data_cube[:, :, b_band])
                ], axis=2)
                fig = plot_rgb_image(
                    rgb_img, 
                    f'三波段合成图（R:{wavelengths[r_band]}nm, G:{wavelengths[g_band]}nm, B:{wavelengths[b_band]}nm）'
                )
                if Config.output_settings["rgb_composite"]:
                    fig.savefig(os.path.join(file_result_dir, f"{base_name}_三波段合成图.png"), dpi=300)
                plt.close()
        else:
            print("[WARN] 无法获取波长数据，跳过三波段合成图")
    else:
        print("[WARN] 波段数不足3，无法生成三波段合成图")


    # MCARI/nMCARI指数计算
    wavelengths = extract_wavelengths(hdr_path)
    if wavelengths is None or len(wavelengths) != bands:
        print("[ERROR] 波长信息无效，无法计算MCARI指数，跳过该文件")
        return
    band_indices = {}
    for name, target_wl in Config.target_wls.items():
        diffs = [abs(wl - target_wl) for wl in wavelengths]
        band_indices[name] = np.argmin(diffs)
        print(f"[INFO] {name}波段：目标{target_wl}nm → 匹配{wavelengths[band_indices[name]]}nm（索引{band_indices[name]}）")

    G = data_cube[:, :, band_indices["G"]]
    R = data_cube[:, :, band_indices["R"]]
    RE = data_cube[:, :, band_indices["RE"]]
    R_safe = np.where(R == 0, 1e-8, R)
    mcari = ((RE - R) - 0.2 * (RE - G)) * (RE / R_safe)
    print(f"[INFO] MCARI指数范围：{np.min(mcari):.4f} ~ {np.max(mcari):.4f}")

    Mmax, Mmin = np.max(mcari), np.min(mcari)
    n_mcari = (mcari - Mmin) / (Mmax - Mmin) if not np.isclose(Mmax, Mmin, atol=1e-10) else np.zeros_like(mcari)

    # 保存MCARI/nMCARI图
    plt.figure(figsize=(10, 8))
    vmin_m, vmax_m = np.percentile(mcari, 1), np.percentile(mcari, 99)
    plt.imshow(mcari, cmap='RdYlGn', vmin=vmin_m, vmax=vmax_m)
    plt.colorbar(label='MCARI指数值', shrink=0.8)
    plt.title(f'MCARI植被指数（{rows}×{cols}）', fontsize=12)
    plt.xlabel('列数', fontsize=10)
    plt.ylabel('行数', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["mcari_index"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_MCARI指数图.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(n_mcari, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='nMCARI指数值（归一化）', shrink=0.8)
    plt.title(f'归一化nMCARI植被指数（{rows}×{cols}）', fontsize=12)
    plt.xlabel('列数', fontsize=10)
    plt.ylabel('行数', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["nmcari_index"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_nMCARI指数图.png"), dpi=300)
    plt.close()


    # 掩膜生成与目标叶片区域提取
    mask_thresholded = n_mcari > Config.threshold
    count_thresholded = np.sum(mask_thresholded)
    print(f"[INFO] 步骤1-阈值分割：前景像素数={count_thresholded}（阈值={Config.threshold}）")
    # 步骤1图不显示图例
    fig = plot_single_mask(mask_thresholded, "步骤1：阈值分割结果", count_thresholded, show_colorbar=False)
    if Config.output_settings["threshold_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_步骤1_阈值分割.png"), dpi=300)
    plt.close()

    def dbscan_denoising_optimized(binary_mask, target_count):
        foreground_coords = np.argwhere(binary_mask)
        if len(foreground_coords) < 10:
            return binary_mask.copy()
        dbscan = DBSCAN(
            eps=Config.dbscan_eps,        
            min_samples=Config.dbscan_min_samples,  
            metric='euclidean'
        )
        labels = dbscan.fit_predict(foreground_coords)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        noise_count = label_counts[unique_labels == -1].sum() if -1 in unique_labels else 0
        valid_labels = unique_labels[unique_labels != -1]
        valid_count = len(valid_labels)
        print(f"[INFO] 步骤2-聚类去噪：总前景={len(foreground_coords)}，噪声点={noise_count}，有效目标叶片数={valid_count}")
        
        # 新增逻辑：当目标数量为0时自动使用所有有效目标，否则使用指定数量
        if target_count <= 0:
            actual_count = valid_count
            print(f"[INFO] 自动模式：将保留所有{actual_count}个有效目标叶片")
        else:
            actual_count = min(target_count, valid_count)
            print(f"[INFO] 指定模式：将保留{actual_count}个最大目标叶片（目标数量：{target_count}，实际有效数：{valid_count}）")
        
        if actual_count <= 0:
            return np.zeros_like(binary_mask, dtype=bool)
        
        label_count_pairs = [(label, count) for label, count in zip(unique_labels, label_counts) if label != -1]
        label_count_pairs.sort(key=lambda x: x[1], reverse=True)
        selected_labels = [pair[0] for pair in label_count_pairs[:actual_count]]
        denoised_mask = np.zeros_like(binary_mask, dtype=bool)
        for label in selected_labels:
            coords = foreground_coords[labels == label]
            denoised_mask[coords[:, 0], coords[:, 1]] = True
        return denoised_mask

    mask_denoised = dbscan_denoising_optimized(mask_thresholded, Config.target_foreground_count)
    count_denoised = np.sum(mask_denoised)
    print(f"[INFO] 步骤2-聚类去噪后：前景像素数={count_denoised}")
    # 步骤2图不显示图例
    fig = plot_single_mask(mask_denoised, "步骤2：优化聚类去噪结果", count_denoised, show_colorbar=False)
    if Config.output_settings["denoised_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_步骤2_聚类去噪.png"), dpi=300)
    plt.close()

    structure = rectangle(*Config.closing_struct_size)
    mask_closed = closing(mask_denoised, structure)
    count_closed = np.sum(mask_closed)
    print(f"[INFO] 步骤3-闭运算后：前景像素数={count_closed}")
    # 步骤3图不显示图例
    fig = plot_single_mask(mask_closed, "步骤3：闭运算补空洞", count_closed, show_colorbar=False)
    if Config.output_settings["closed_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_步骤3_闭运算.png"), dpi=300)
    plt.close()

    mask_filled = remove_small_holes(
        mask_closed,
        area_threshold=Config.hole_threshold,
        connectivity=2
    )
    count_filled = np.sum(mask_filled)
    print(f"[INFO] 步骤4-补全空洞后：前景像素数={count_filled}")
    # 步骤4图不显示图例
    fig = plot_single_mask(mask_filled, "步骤4：补全内部空洞", count_filled, show_colorbar=False)
    if Config.output_settings["filled_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_步骤4_补全空洞.png"), dpi=300)
    plt.close()

    min_size = max(Config.min_object_size, count_denoised // 10)
    mask_final = remove_small_objects(mask_filled, min_size=min_size)
    count_final = np.sum(mask_final)
    print(f"[INFO] 步骤5-最终目标叶片区域：前景像素数={count_final}")
    # 步骤5图显示图例
    fig = plot_single_mask(mask_final, "步骤5：最终目标叶片区域", count_final, show_colorbar=False)
    if Config.output_settings["final_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_步骤5_最终目标叶片区域.png"), dpi=300)
    plt.close()


    # 生成目标叶片区域掩膜叠加图用于全流程对比
    if rgb_img is not None:
        base_img = rgb_img
    else:
        base_img = n_mcari
        
    mask_overlay = np.zeros_like(base_img)
    if base_img.ndim == 3:
        mask_overlay[mask_final] = [1, 0, 0]  # 红色高亮目标叶片
    else:
        mask_overlay = mask_final.astype(float)
        
    overlay_img = np.copy(base_img)
    if base_img.ndim == 3:
        overlay_img[mask_final] = overlay_img[mask_final] * 0.5 + np.array([1, 0, 0]) * 0.5
    else:
        overlay_img = np.stack([overlay_img, overlay_img, overlay_img], axis=2)
        overlay_img[mask_final] = overlay_img[mask_final] * 0.5 + np.array([1, 0, 0]) * 0.5


    # 全流程对比图，增加掩膜叠加图作为第6步
    plt.figure(figsize=(20, 12))
    steps = [
        (mask_thresholded, f"1. 阈值分割（{Config.threshold}）"),
        (mask_denoised, f"2. 优化聚类去噪（{'自动判断数量' if Config.target_foreground_count == 0 else f'保留{Config.target_foreground_count}个目标叶片'}）"),
        (mask_closed, f"3. 闭运算补洞"),
        (mask_filled, f"4. 补全空洞"),
        (mask_final, f"5. 最终目标叶片区域"),
        (overlay_img, "6. 目标叶片区域掩膜叠加")
    ]
    for i, (mask, title) in enumerate(steps, 1):
        plt.subplot(2, 3, i)  # 修改为3行2列布局以容纳6张图
        plt.imshow(mask, cmap='gray' if i < 6 else None)
        plt.title(title, fontsize=20)
        plt.axis('off')
    plt.tight_layout()
    if Config.output_settings["process_comparison"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_全流程对比图.png"), dpi=300)
    plt.close()


    # 最终掩膜叠加（红色高亮目标叶片）
    if rgb_img is not None:
        base_img = rgb_img
        base_title = "三波段合成图"
    else:
        base_img = n_mcari
        base_title = "nMCARI指数图"

    mask_overlay = np.zeros_like(base_img)
    if base_img.ndim == 3:
        mask_overlay[mask_final] = [1, 0, 0]  # 红色高亮目标叶片
    else:
        mask_overlay = mask_final.astype(float)

    plt.figure(figsize=(12, 10))
    plt.imshow(base_img)
    plt.imshow(mask_overlay, alpha=0.5)
    plt.title(f'目标叶片区域掩膜叠加（红色为目标叶片，底图：{base_title}）', fontsize=14)
    plt.xlabel('列数', fontsize=10)
    plt.ylabel('行数', fontsize=10)
    if base_img.ndim == 2:
        plt.colorbar(label='底图像素值')
    plt.tight_layout()
    if Config.output_settings["mask_overlay"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_目标叶片区域掩膜叠加.png"), dpi=300)
    plt.close()

    foreground_only = base_img.copy()
    if base_img.ndim == 3:
        foreground_only[~mask_final] = [0, 0, 0]
    else:
        foreground_only[~mask_final] = 0
    
    plt.figure(figsize=(12, 10))
    plt.imshow(foreground_only)
    plt.title(f'仅保留目标叶片区域（底图：{base_title}）', fontsize=14)
    plt.xlabel('列数', fontsize=10)
    plt.ylabel('行数', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["foreground_only"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_仅保留目标叶片区域.png"), dpi=300)
    plt.close()
    
    
    # 目标叶片波段反射率提取与数据框输出
    labeled_mask = label(mask_final, connectivity=2)
    n_targets = labeled_mask.max()  # 目标叶片数量

    if n_targets == 0:
        print("[WARN] 未检测到有效目标叶片，无法提取反射率数据")
    else:
        print(f"[INFO] 检测到 {n_targets} 个目标叶片，开始提取波段反射率...")
        wavelengths = extract_wavelengths(hdr_path)
        if wavelengths is not None:
            target_data = {f"目标叶片_{i+1}": [] for i in range(n_targets)}
            
            for band_idx in range(bands):
                band_data = data_cube[:, :, band_idx]
                for target_id in range(1, n_targets + 1):
                    target_pixels = band_data[labeled_mask == target_id]
                    mean_reflectance = np.mean(target_pixels) if len(target_pixels) > 0 else np.nan
                    target_data[f"目标叶片_{target_id}"].append(mean_reflectance)
            
            target_data["波段波长(nm)"] = wavelengths
            columns_order = ["波段波长(nm)"] + [f"目标叶片_{i+1}" for i in range(n_targets)]
            reflectance_df = pd.DataFrame(target_data, columns=columns_order)
            
            print("\n[INFO] 波段反射率数据框（前5行）：")
            print(reflectance_df.head())
            print(f"\n[INFO] 数据框形状：{reflectance_df.shape}（{reflectance_df.shape[0]}个波段，{reflectance_df.shape[1]-1}个目标叶片）")
            
            if Config.output_settings["reflectance_data"]:
                output_filename = os.path.join(file_result_dir, f"{base_name}_目标叶片反射率数据.csv")
                reflectance_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
                print(f"[INFO] 反射率数据已保存至：{output_filename}")


    # 目标叶片编号标记与保存
    if 'labeled_mask' in locals() and n_targets > 0:
        target_props = regionprops(labeled_mask)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(labeled_mask, cmap='tab20', alpha=0.7)
        if rgb_img is not None:
            plt.imshow(rgb_img, alpha=0.3)
        else:
            plt.imshow(n_mcari, alpha=0.3, cmap='gray')
        
        for i, prop in enumerate(target_props, 1):
            centroid = (int(prop.centroid[1]), int(prop.centroid[0]))
            plt.text(
                centroid[0], centroid[1], 
                str(i), 
                color='white', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='red', edgecolor='black', pad=5, alpha=0.8)
            )
        
        plt.title(f'目标叶片标记（共{len(target_props)}个区域）', fontsize=14)
        plt.xlabel('列数', fontsize=10)
        plt.ylabel('行数', fontsize=10)
        plt.tight_layout()
        
        if Config.output_settings["target_labeling"]:
            img_filename = os.path.join(file_result_dir, f"{base_name}_目标叶片标记图.png")
            plt.savefig(img_filename, dpi=300, bbox_inches='tight')
            print(f"[INFO] 目标叶片标记图已保存至：{img_filename}")
        plt.close()
    else:
        print("[WARN] 无有效目标叶片，跳过标记图生成")
    
    print(f"[INFO] 文件 {base_name} 处理完成\n")


# 主函数
def main():
    start_time = time.time()  # 记录开始时间
    os.makedirs(Config.result_dir, exist_ok=True)
    file_pairs = get_file_pairs(Config.data_dir)
    
    if not file_pairs:
        print("[INFO] 未找到有效的文件对，程序退出")
        end_time = time.time()
        print(f"[INFO] 总运行时间: {end_time - start_time:.2f} 秒")
        return
    
    print(f"[INFO] 发现 {len(file_pairs)} 个有效的文件对，开始批量处理...")
    
    for base_name, hdr_path, float_path in file_pairs:
        process_single_file(base_name, hdr_path, float_path, Config.result_dir)
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"[INFO] 所有文件处理完成，结果保存在 {Config.result_dir} 目录下")
    print(f"[INFO] 运行时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()