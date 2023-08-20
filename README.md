# K-Means Image Quantization

This repository contains a Python script that implements the K-Means clustering algorithm for image quantization. Image quantization involves reducing the number of colors in an image while preserving its visual content. The K-Means algorithm groups similar colors together and assigns representative colors, creating a 'quantized' version of the original image.

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
