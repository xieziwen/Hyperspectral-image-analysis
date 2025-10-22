"""
Created on Wed Oct 22 09:11:41 2025

@author: Ziwen_Xie
"""
# --------------------------
# 1. Import Dependencies
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
# 2. Global Configuration
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
    target_foreground_count = 0  # Number of target leaves, 0 for auto-detection, >0 for specified number
    separate_folder = False  # Whether to place each image result in an independent folder
    
    # New: Parameters to control whether to output results of each step (True=output, False=no output)
    output_settings = {
        "single_band": False,        # Single band image
        "rgb_composite": True,      # Three-band composite image
        "mcari_index": False,        # MCARI index image
        "nmcari_index": False,       # Normalized MCARI index image
        "threshold_mask": False,     # Threshold segmentation result
        "denoised_mask": False,      # Clustering denoising result
        "closed_mask": False,        # Closing operation result
        "filled_mask": False,        # Hole filling result
        "final_mask": False,         # Final target leaf region
        "process_comparison": True, # Full process comparison chart
        "mask_overlay": False,       # Target leaf region mask overlay
        "foreground_only": True,    # Only retain target leaf regions
        "reflectance_data": True,   # Reflectance data CSV
        "target_labeling": True     # Target leaf labeling map
    }

# Filter matplotlib warnings about missing fonts
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
# New: Filter specific warnings related to font finding
warnings.filterwarnings("ignore", message="findfont: Font family.*not found.")

# Set commonly available English fonts for the system (prioritize Arial, Helvetica)

warnings.filterwarnings("ignore", message="findfont: Font family 'Helvetica' not found.")
warnings.filterwarnings("ignore", message="findfont: Font family.*not found.") 
plt.rcParams["axes.unicode_minus"] = False  # Solve the problem of negative sign display

# --------------------------
# 3. Utility Functions
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
        print(f"[ERROR] Failed to extract hdr parameters: {str(e)}")
        return None, None, None


def extract_wavelengths(hdr_path):
    try:
        with open(hdr_path, 'r') as f:
            content = f.read()
        wave_match = re.search(r'wavelength\s*=\s*\{(.*?)\}', content, re.IGNORECASE | re.DOTALL)
        if not wave_match:
            print("[ERROR] Wavelength field not found in hdr file")
            return None
        wave_str = wave_match.group(1).strip().replace('\n', '').replace(' ', '')
        wavelengths = list(map(float, wave_str.split(',')))
        return wavelengths
    except Exception as e:
        print(f"[ERROR] Failed to extract wavelengths: {str(e)}")
        return None


def plot_single_mask(mask, title, pixel_count, figsize=(10, 8), show_colorbar=True):
    plt.figure(figsize=figsize)
    plt.imshow(mask, cmap='gray')
    if show_colorbar:
        plt.colorbar(label='Pixel Value (1=Foreground, 0=Background)')
    plt.title(f'{title} (Foreground Pixels={pixel_count})', fontsize=12)
    plt.xlabel('Columns', fontsize=10)
    plt.ylabel('Rows', fontsize=10)
    plt.tight_layout()
    return plt


def plot_rgb_image(rgb_img, title, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(rgb_img)
    plt.title(title, fontsize=12)
    plt.xlabel(f'Columns ({rgb_img.shape[1]})', fontsize=10)
    plt.ylabel(f'Rows ({rgb_img.shape[0]})', fontsize=10)
    plt.tight_layout()
    return plt


def get_file_pairs(data_dir):
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory {data_dir} does not exist")
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
            print(f"[WARN] Skipping incomplete file pair: {base_name} (missing .hdr or .float file)")
    return valid_pairs


def process_single_file(base_name, hdr_path, float_path, result_dir):
    print(f"\n[INFO] Starting to process file: {base_name}")
    
    # Determine result save directory based on configuration
    if Config.separate_folder:
        file_result_dir = os.path.join(result_dir, base_name)
    else:
        file_result_dir = result_dir
    os.makedirs(file_result_dir, exist_ok=True)
    
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial", "sans-serif"] 
    plt.rcParams["axes.unicode_minus"] = False

    # Read hdr parameters
    lines_hdr, samples_hdr, bands_hdr = extract_hdr_params(hdr_path)
    if None in [lines_hdr, samples_hdr, bands_hdr]:
        print(f"[ERROR] Failed to extract complete dimension information from {base_name}.hdr, skipping this file")
        return
    rows, cols, bands = lines_hdr, samples_hdr, bands_hdr
    print(f"[INFO] Image parameters: Rows={rows}, Columns={cols}, Bands={bands}")

    # Read hyperspectral data
    try:
        img = envi.open(hdr_path, float_path)
        data_cube = img.load()
        total_elements = rows * cols * bands
        if data_cube.size != total_elements:
            raise ValueError(f"Data size mismatch (actual {data_cube.size}, expected {total_elements})")
        data_cube = data_cube.reshape((rows, cols, bands))
        print(f"[INFO] Reshaped data shape: {data_cube.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to read data: {str(e)}, skipping this file")
        return

    # Save single band image
    band_to_show = min(49, bands - 1)
    band_data = np.squeeze(data_cube[:, :, band_to_show])
    plt.figure(figsize=(10, 8))
    plt.imshow(band_data, cmap='gray', vmin=np.percentile(band_data, 1), vmax=np.percentile(band_data, 99))
    plt.colorbar(label='Reflectance Value')
    plt.title(f'Hyperspectral Single Band Image ({cols}×{rows}) - Band {band_to_show+1}', fontsize=12)
    plt.xlabel(f'Columns ({cols})', fontsize=10)
    plt.ylabel(f'Rows ({rows})', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["single_band"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_single_band_image.png"), dpi=300)
    plt.close()

    # Three-band composite image (target wavelength matching)
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
                    print(f"[INFO] {channel} channel: Target {target_wl}nm → Matched {wavelengths[idx]}nm (Index {idx})")
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
                    f'Three-Band Composite Image (R:{wavelengths[r_band]}nm, G:{wavelengths[g_band]}nm, B:{wavelengths[b_band]}nm)'
                )
                if Config.output_settings["rgb_composite"]:
                    fig.savefig(os.path.join(file_result_dir, f"{base_name}_rgb_composite.png"), dpi=300)
                plt.close()
        else:
            print("[WARN] Unable to obtain wavelength data, skipping three-band composite image")
    else:
        print("[WARN] Insufficient number of bands (needs 3), cannot generate three-band composite image")


    # MCARI/nMCARI index calculation
    wavelengths = extract_wavelengths(hdr_path)
    if wavelengths is None or len(wavelengths) != bands:
        print("[ERROR] Invalid wavelength information, cannot calculate MCARI index, skipping this file")
        return
    band_indices = {}
    for name, target_wl in Config.target_wls.items():
        diffs = [abs(wl - target_wl) for wl in wavelengths]
        band_indices[name] = np.argmin(diffs)
        print(f"[INFO] {name} band: Target {target_wl}nm → Matched {wavelengths[band_indices[name]]}nm (Index {band_indices[name]})")

    G = data_cube[:, :, band_indices["G"]]
    R = data_cube[:, :, band_indices["R"]]
    RE = data_cube[:, :, band_indices["RE"]]
    R_safe = np.where(R == 0, 1e-8, R)
    mcari = ((RE - R) - 0.2 * (RE - G)) * (RE / R_safe)
    print(f"[INFO] MCARI index range: {np.min(mcari):.4f} ~ {np.max(mcari):.4f}")

    Mmax, Mmin = np.max(mcari), np.min(mcari)
    n_mcari = (mcari - Mmin) / (Mmax - Mmin) if not np.isclose(Mmax, Mmin, atol=1e-10) else np.zeros_like(mcari)

    # Save MCARI/nMCARI images
    plt.figure(figsize=(10, 8))
    vmin_m, vmax_m = np.percentile(mcari, 1), np.percentile(mcari, 99)
    plt.imshow(mcari, cmap='RdYlGn', vmin=vmin_m, vmax=vmax_m)
    plt.colorbar(label='MCARI Index Value', shrink=0.8)
    plt.title(f'MCARI Vegetation Index ({rows}×{cols})', fontsize=12)
    plt.xlabel('Columns', fontsize=10)
    plt.ylabel('Rows', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["mcari_index"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_mcari_index.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(n_mcari, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='nMCARI Index Value (Normalized)', shrink=0.8)
    plt.title(f'Normalized nMCARI Vegetation Index ({rows}×{cols})', fontsize=12)
    plt.xlabel('Columns', fontsize=10)
    plt.ylabel('Rows', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["nmcari_index"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_nmcari_index.png"), dpi=300)
    plt.close()


    # Mask generation and target leaf region extraction
    mask_thresholded = n_mcari > Config.threshold
    count_thresholded = np.sum(mask_thresholded)
    print(f"[INFO] Step 1-Threshold Segmentation: Foreground pixels={count_thresholded} (Threshold={Config.threshold})")
    # No colorbar for Step 1 image
    fig = plot_single_mask(mask_thresholded, "Step 1: Threshold Segmentation Result", count_thresholded, show_colorbar=False)
    if Config.output_settings["threshold_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_step1_threshold_segmentation.png"), dpi=300)
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
        print(f"[INFO] Step 2-Clustering Denoising: Total foreground={len(foreground_coords)}, Noise points={noise_count}, Valid target leaves={valid_count}")
        
        # New logic: Use all valid targets when target count is 0, otherwise use specified count
        if target_count <= 0:
            actual_count = valid_count
            print(f"[INFO] Automatic mode: Will retain all {actual_count} valid target leaves")
        else:
            actual_count = min(target_count, valid_count)
            print(f"[INFO] Specified mode: Will retain {actual_count} largest target leaves (Target count: {target_count}, Actual valid count: {valid_count})")
        
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
    print(f"[INFO] Step 2-After Clustering Denoising: Foreground pixels={count_denoised}")
    # No colorbar for Step 2 image
    fig = plot_single_mask(mask_denoised, "Step 2: Optimized Clustering Denoising Result", count_denoised, show_colorbar=False)
    if Config.output_settings["denoised_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_step2_clustering_denoising.png"), dpi=300)
    plt.close()

    structure = rectangle(*Config.closing_struct_size)
    mask_closed = closing(mask_denoised, structure)
    count_closed = np.sum(mask_closed)
    print(f"[INFO] Step 3-After Closing Operation: Foreground pixels={count_closed}")
    # No colorbar for Step 3 image
    fig = plot_single_mask(mask_closed, "Step 3: Closing Operation for Hole Filling", count_closed, show_colorbar=False)
    if Config.output_settings["closed_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_step3_closing_operation.png"), dpi=300)
    plt.close()

    mask_filled = remove_small_holes(
        mask_closed,
        area_threshold=Config.hole_threshold,
        connectivity=2
    )
    count_filled = np.sum(mask_filled)
    print(f"[INFO] Step 4-After Hole Filling: Foreground pixels={count_filled}")
    # No colorbar for Step 4 image
    fig = plot_single_mask(mask_filled, "Step 4: Internal Hole Filling", count_filled, show_colorbar=False)
    if Config.output_settings["filled_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_step4_hole_filling.png"), dpi=300)
    plt.close()

    min_size = max(Config.min_object_size, count_denoised // 10)
    mask_final = remove_small_objects(mask_filled, min_size=min_size)
    count_final = np.sum(mask_final)
    print(f"[INFO] Step 5-Final Target Leaf Region: Foreground pixels={count_final}")
    # Show colorbar for Step 5 image
    fig = plot_single_mask(mask_final, "Step 5: Final Target Leaf Region", count_final, show_colorbar=False)
    if Config.output_settings["final_mask"]:
        fig.savefig(os.path.join(file_result_dir, f"{base_name}_step5_final_target_region.png"), dpi=300)
    plt.close()


    # Generate target leaf region mask overlay for full process comparison
    if rgb_img is not None:
        base_img = rgb_img
    else:
        base_img = n_mcari
        
    mask_overlay = np.zeros_like(base_img)
    if base_img.ndim == 3:
        mask_overlay[mask_final] = [1, 0, 0]  # Red highlight for target leaves
    else:
        mask_overlay = mask_final.astype(float)
        
    overlay_img = np.copy(base_img)
    if base_img.ndim == 3:
        overlay_img[mask_final] = overlay_img[mask_final] * 0.5 + np.array([1, 0, 0]) * 0.5
    else:
        overlay_img = np.stack([overlay_img, overlay_img, overlay_img], axis=2)
        overlay_img[mask_final] = overlay_img[mask_final] * 0.5 + np.array([1, 0, 0]) * 0.5


    # Full process comparison chart, add mask overlay as Step 6
    plt.figure(figsize=(20, 12))
    steps = [
        (mask_thresholded, f"1. Threshold Segmentation ({Config.threshold})"),
        (mask_denoised, f"2. Optimized Clustering Denoising ({'Auto-detect count' if Config.target_foreground_count == 0 else f'Retain {Config.target_foreground_count} target leaves'})"),
        (mask_closed, f"3. Closing Operation for Hole Filling"),
        (mask_filled, f"4. Hole Filling"),
        (mask_final, f"5. Final Target Leaf Region"),
        (overlay_img, "6. Target Leaf Region Mask Overlay")
    ]
    for i, (mask, title) in enumerate(steps, 1):
        plt.subplot(2, 3, i)  # Modified to 2 rows x 3 columns layout to accommodate 6 images
        plt.imshow(mask, cmap='gray' if i < 6 else None)
        plt.title(title, fontsize=20)
        plt.axis('off')
    plt.tight_layout()
    if Config.output_settings["process_comparison"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_full_process_comparison.png"), dpi=300)
    plt.close()


    # Final mask overlay (red highlight for target leaves)
    if rgb_img is not None:
        base_img = rgb_img
        base_title = "Three-band Composite Image"
    else:
        base_img = n_mcari
        base_title = "nMCARI Index Image"

    mask_overlay = np.zeros_like(base_img)
    if base_img.ndim == 3:
        mask_overlay[mask_final] = [1, 0, 0]  # Red highlight for target leaves
    else:
        mask_overlay = mask_final.astype(float)

    plt.figure(figsize=(12, 10))
    plt.imshow(base_img)
    plt.imshow(mask_overlay, alpha=0.5)
    plt.title(f'Target Leaf Region Mask Overlay (Red = Target Leaves, Base Image: {base_title})', fontsize=14)
    plt.xlabel('Columns', fontsize=10)
    plt.ylabel('Rows', fontsize=10)
    if base_img.ndim == 2:
        plt.colorbar(label='Base Image Pixel Value')
    plt.tight_layout()
    if Config.output_settings["mask_overlay"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_target_region_overlay.png"), dpi=300)
    plt.close()

    foreground_only = base_img.copy()
    if base_img.ndim == 3:
        foreground_only[~mask_final] = [0, 0, 0]
    else:
        foreground_only[~mask_final] = 0
    
    plt.figure(figsize=(12, 10))
    plt.imshow(foreground_only)
    plt.title(f'Only Target Leaf Regions Retained (Base Image: {base_title})', fontsize=14)
    plt.xlabel('Columns', fontsize=10)
    plt.ylabel('Rows', fontsize=10)
    plt.tight_layout()
    if Config.output_settings["foreground_only"]:
        plt.savefig(os.path.join(file_result_dir, f"{base_name}_only_target_regions.png"), dpi=300)
    plt.close()
    
    
    # Target leaf band reflectance extraction and data frame output
    labeled_mask = label(mask_final, connectivity=2)
    n_targets = labeled_mask.max()  # Number of target leaves

    if n_targets == 0:
        print("[WARN] No valid target leaves detected, cannot extract reflectance data")
    else:
        print(f"[INFO] Detected {n_targets} target leaves, starting to extract band reflectance...")
        wavelengths = extract_wavelengths(hdr_path)
        if wavelengths is not None:
            target_data = {f"Target_Leaf_{i+1}": [] for i in range(n_targets)}
            
            for band_idx in range(bands):
                band_data = data_cube[:, :, band_idx]
                for target_id in range(1, n_targets + 1):
                    target_pixels = band_data[labeled_mask == target_id]
                    mean_reflectance = np.mean(target_pixels) if len(target_pixels) > 0 else np.nan
                    target_data[f"Target_Leaf_{target_id}"].append(mean_reflectance)
            
            target_data["Band_Wavelength(nm)"] = wavelengths
            columns_order = ["Band_Wavelength(nm)"] + [f"Target_Leaf_{i+1}" for i in range(n_targets)]
            reflectance_df = pd.DataFrame(target_data, columns=columns_order)
            
            print("\n[INFO] Band reflectance data frame (first 5 rows):")
            print(reflectance_df.head())
            print(f"\n[INFO] Data frame shape: {reflectance_df.shape} ({reflectance_df.shape[0]} bands, {reflectance_df.shape[1]-1} target leaves)")
            
            if Config.output_settings["reflectance_data"]:
                output_filename = os.path.join(file_result_dir, f"{base_name}_target_leaf_reflectance_data.csv")
                reflectance_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
                print(f"[INFO] Reflectance data saved to: {output_filename}")


    # Target leaf labeling and saving
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
        
        plt.title(f'Target Leaf Labeling (Total {len(target_props)} Regions)', fontsize=14)
        plt.xlabel('Columns', fontsize=10)
        plt.ylabel('Rows', fontsize=10)
        plt.tight_layout()
        
        if Config.output_settings["target_labeling"]:
            img_filename = os.path.join(file_result_dir, f"{base_name}_target_leaf_labeling.png")
            plt.savefig(img_filename, dpi=300, bbox_inches='tight')
            print(f"[INFO] Target leaf labeling image saved to: {img_filename}")
        plt.close()
    else:
        print("[WARN] No valid target leaves, skipping labeling image generation")
    
    print(f"[INFO] File {base_name} processing completed\n")


# Main function
def main():
    start_time = time.time()  # Record start time
    os.makedirs(Config.result_dir, exist_ok=True)
    file_pairs = get_file_pairs(Config.data_dir)
    
    if not file_pairs:
        print("[INFO] No valid file pairs found, program exiting")
        end_time = time.time()
        print(f"[INFO] Total running time: {end_time - start_time:.2f} seconds")
        return
    
    print(f"[INFO] Found {len(file_pairs)} valid file pairs, starting batch processing...")
    
    for base_name, hdr_path, float_path in file_pairs:
        process_single_file(base_name, hdr_path, float_path, Config.result_dir)
    
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"[INFO] All files processed, results saved in {Config.result_dir} directory")
    print(f"[INFO] Running time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()