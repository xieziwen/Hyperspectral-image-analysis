# SOC710 High Spectral Leaf Extraction Batch Processing Tool

## Overview
The SOC710_HS_extract_batch.py is a Python tool designed for batch processing of SOC 710 hyperspectral images, specifically focused on extracting target leaf regions and analyzing their spectral characteristics. This tool implements a complete workflow from image loading, target region segmentation based on vegetation index threshold, band reflectance extraction and result visualization, providing researchers with efficient hyperspectral data analysis capabilities for plant leaf studies.

## Features
- Batch processing of hyperspectral image pairs (.hdr + .float)
- Multi-step image processing pipeline: threshold segmentation, DBSCAN clustering denoising, morphological operations
- Vegetation index calculation (MCARI and normalized MCARI)
- Customizable processing parameters via configuration class
- Multiple visualization outputs including RGB composites, index maps, and processing result comparisons
- Extraction and export of average reflectance data for target leaf regions
- Automatic labeling and counting of target leaf regions

## Dependencies
The following Python libraries are required:
- numpy
- matplotlib
- spectral (spectral-python)
- scikit-learn
- scikit-image
- pandas
- os, re (built-in)

Install dependencies using pip:
```bash
pip install numpy matplotlib spectral scikit-learn scikit-image pandas
```

## Usage
1. Prepare your hyperspectral image data (.hdr and .float file pairs) and place them in the data directory
2. Adjust configuration parameters in the `Config` class if needed (see Configuration section)
3. Run the script:
```bash
python SOC710_HS_extract_batch_CHI.py
```
or
```
python SOC710_HS_extract_batch_ENG.py
```
4. Check processing results in the specified result directory

## Configuration Parameters
The `Config` class contains adjustable parameters to control the processing workflow:

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Directory containing input hyperspectral files |
| `result_dir` | Directory for saving processing results |
| `target_wls` | Target wavelengths for vegetation index calculation |
| `threshold` | Threshold for initial segmentation |
| `dbscan_eps`, `dbscan_min_samples` | DBSCAN clustering parameters for denoising |
| `closing_struct_size` | Structure size for morphological closing operation |
| `hole_threshold` | Area threshold for removing small holes |
| `min_object_size` | Minimum size threshold for target regions |
| `target_rgb_wavelengths` | Target wavelengths for RGB composite image |
| `target_foreground_count` | Number of target leaves to retain (0 for automatic detection) |
| `separate_folder` | Whether to save results in separate folders per image |
| `output_settings` | Dictionary controlling which results to output |

## Output Results
The tool generates various output files based on configuration:

1. **Image Files**:
   - Single band images
   - RGB composite images
   - MCARI and normalized MCARI index maps
   - Processing step visualizations (thresholding, denoising, morphological operations)
   - Final target leaf region masks
   - Overlay images of target regions on original images
   - Target leaf labeling maps

2. **Data Files**:
   - CSV files containing average reflectance data for each detected leaf across all wavelengths

## Processing Workflow
1. **Data Loading**: Reads hyperspectral image data and header files
2. **Preprocessing**: Generates single band and RGB composite images
3. **Index Calculation**: Computes MCARI and normalized MCARI vegetation indices
4. **Segmentation**: Applies thresholding to separate foreground (leaves) from background
5. **Denoising**: Uses DBSCAN clustering to remove noise and select target regions
6. **Morphological Operations**: Applies closing and hole filling to improve region quality
7. **Post-processing**: Removes small objects and refines target regions
8. **Analysis**: Extracts and saves reflectance data for each detected leaf
9. **Visualization**: Generates and saves various result visualizations

## Notes
- Ensure .hdr and .float files form valid pairs with the same base name
- The tool handles basic errors but may require parameter adjustment for different image conditions
- Processing time depends on image size and number of files; large datasets may take longer
- For best results, adjust threshold values and morphological parameters based on specific imaging conditions

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SOC710高光谱叶片提取批量处理工具

## 概述
SOC710_HS_extract_batch.py是一个用于批量处理SOC 710高光谱图像的Python工具，专门用于提取目标叶片区域并分析其光谱特征。该工具实现了从图像加载、基于植被指数阈值的目标区域分割、波段反射率提取和结果可视化的完整工作流程，为研究人员提供高效的植物叶片高光谱数据分析能力。

## 功能特点
- 批量处理高光谱图像对（.hdr + .float）
- 多步骤图像处理流程：阈值分割、DBSCAN聚类去噪、形态学操作
- 植被指数计算（MCARI和归一化MCARI）
- 通过配置类自定义处理参数
- 多种可视化输出，包括RGB合成图、指数图和处理结果对比图
- 提取并导出目标叶片区域的平均反射率数据
- 自动标记和计数目标叶片区域

## 依赖库
需要以下Python库：
- numpy
- matplotlib
- spectral (spectral-python)
- scikit-learn
- scikit-image
- pandas
- os, re（内置库）

使用pip安装依赖：
```bash
pip install numpy matplotlib spectral scikit-learn scikit-image pandas
```

## 使用方法
1. 准备高光谱图像数据（.hdr和.float文件）并放置在data目录中
2. 根据需要调整`Config`类中的配置参数（参见配置部分）
3. 运行脚本：
```bash
python SOC710_HS_extract_batch_CHI.py
```
或者
```
python SOC710_HS_extract_batch_ENG.py
```
4. 在指定的结果目录中查看处理结果

## 配置参数
`Config`类包含可调整的参数，用于控制处理流程：

| 参数 | 描述 |
|------|------|
| `data_dir` | 包含输入高光谱文件的目录 |
| `result_dir` | 用于保存处理结果的目录 |
| `target_wls` | 用于植被指数计算的目标波长 |
| `threshold` | 初始分割的阈值 |
| `dbscan_eps`, `dbscan_min_samples` | 用于去噪的DBSCAN聚类参数 |
| `closing_struct_size` | 形态学闭运算的结构尺寸 |
| `hole_threshold` | 用于去除小孔洞的面积阈值 |
| `min_object_size` | 目标区域的最小尺寸阈值 |
| `target_rgb_wavelengths` | 用于RGB合成图像的目标波长 |
| `target_foreground_count` | 要保留的目标叶片数量（0表示自动检测） |
| `separate_folder` | 是否将结果保存在每个图像的独立文件夹中 |
| `output_settings` | 控制输出哪些结果的字典 |

## 输出结果
该工具根据配置生成各种输出文件：

1. **图像文件**：
   - 单波段图像
   - RGB合成图像
   - MCARI和归一化MCARI指数图
   - 处理步骤可视化（阈值分割、去噪、形态学操作）
   - 最终目标叶片区域掩膜
   - 目标区域在原始图像上的叠加图
   - 目标叶片标记图

2. **数据文件**：
   - 包含每个检测到的叶片在所有波长上的平均反射率数据的CSV文件

## 处理流程
1. **数据加载**：读取高光谱图像数据和头文件
2. **预处理**：生成单波段和RGB合成图像
3. **指数计算**：计算MCARI和归一化MCARI植被指数
4. **分割**：应用阈值分割将前景（叶片）与背景分离
5. **去噪**：使用DBSCAN聚类去除噪声并选择目标区域
6. **形态学操作**：应用闭运算和孔洞填充以改善区域质量
7. **后处理**：移除小物体并优化目标区域
8. **分析**：提取并保存每个检测到的叶片的反射率数据
9. **可视化**：生成并保存各种结果可视化

## 注意事项
- 确保存在配套的有效的.hdr和.float文件，具有相同的文件名
- 该工具在大多数情况下基本可行，但可能需要根据不同的图像条件调整参数
- 处理时间取决于图像大小和文件数量；大型数据集可能需要更长时间
- 为获得最佳结果，请根据特定的成像条件调整阈值和形态学参数
