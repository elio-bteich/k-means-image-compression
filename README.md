# K-Means Image Compression

This repository contains a Python script that utilizes the K-Means clustering algorithm for image compression. Image compression involves reducing the data size of an image while striving to maintain its visual quality. The K-Means algorithm groups similar colors together and assigns representative colors, effectively compressing the image's color palette and creating a compressed version of the original image.

The K-Means Image Compression script reads in a JPG image and applies the K-Means clustering algorithm to compress the image's color palette. By reducing the number of distinct colors in the image, it aims to achieve a smaller file size while still preserving the overall visual appearance.

Please note that this script is specifically designed for JPG images.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Prerequisites

To run this script, you need to have the following installed:

- Python (>= 3.6)
- NumPy
- Matplotlib

You can install the required dependencies using the following command:

```bash
pip install numpy matplotlib
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/elio-bteich/k-means-image-quantization.git
```

2. Navigate to the repository directory:

```bash
cd k-means-image-quantization
```

3. Run the script:

```bash
python kmeans_image_compression.py
```

4. Follow the prompts to provide the input image path, number of centroids, and number of iterations.

## Customization

You can customize the behavior of the script by modifying the following parameters:

- `image_path`: Path to the input image.
- `num_centroids`: Number of centroids (colors) to use for quantization.
- `num_iterations`: Number of iterations for the K-Means algorithm.

Feel free to experiment with different values to achieve your desired image quantization results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
