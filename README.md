# JPEG Compression for Astronaut's Moon Images

## Overview
This project implements **JPEG compression using the Discrete Cosine Transform (DCT)** to efficiently reduce the size of high-resolution images captured by an astronaut on the Moon. The compressed images enable faster transmission to the space station while maintaining critical visual details.

## Features
- **RGB to YCbCr Conversion**: Enhances compression efficiency by prioritizing luminance over chrominance.
- **DCT-Based Compression**: Reduces image size using block-wise DCT transformation.
- **Inverse DCT for Reconstruction**: Recovers the original image from the compressed form.
- **Compression Ratio & PSNR Analysis**: Evaluates the trade-off between image quality and compression efficiency.
- **Block Size Comparison**: Analyzes the effect of 8×8, 16×16, and 32×32 block sizes on compression and quality.

## Installation
Ensure you have **MATLAB** installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/jpeg-compression.git
   cd jpeg-compression
   ```
2. Run MATLAB and navigate to the project directory.
3. Execute the script:
   ```matlab
   main
   ```

## Usage
- **Input**: A high-resolution RGB image.
- **Output**: Displays the original, compressed, and reconstructed images along with PSNR and compression ratio analysis.
- **Comparison**: The effect of different block sizes on compression and image quality is visualized.

## Results
- **Compression Ratio**: Remains stable across block sizes.
- **PSNR (Image Quality)**:
  - **8×8 blocks**: Balanced compression and detail.
  - **16×16 blocks**: Slightly reduced quality.
  - **32×32 blocks**: Higher PSNR, better image quality, minor artifacts.
- **Conclusion**: 32×32 blocks offer better image quality while ensuring efficient transmission.

