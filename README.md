# Principal Component Analysis (PCA) on Dog Images

This repository contains an implementation of Principal Component Analysis (PCA) on a dataset of grayscale dog images. The code extracts principal components from the images and visualizes both the principal components and the reconstruction of the images using varying numbers of principal components.

### Key Features
- **Image Loading and Flattening**:
  - Images are loaded from the `q1_dog/` directory, converted to grayscale, and flattened into 1D vectors (3600 features per image).
  
- **Normalization and Covariance Matrix**:
  - The images are centered by subtracting the mean and normalized by dividing by the standard deviation.
  - A covariance matrix is computed based on the normalized data to analyze variance across the image pixels.

- **Eigenvalue and Eigenvector Computation**:
  - Eigenvalues and eigenvectors are computed from the covariance matrix.
  - Eigenvectors are sorted in descending order by their corresponding eigenvalues to capture the directions of maximum variance in the data.

- **Principal Component Visualization**:
  - The top 10 principal components are reshaped back into 60x60 pixel images and visualized to show the major patterns captured from the data.

- **Image Reconstruction**:
  - Images are reconstructed using different numbers of principal components (e.g., k=1, 5, 10, 20, 50, 500, 3600) to show how much of the original image can be approximated with varying levels of information.

### Output
- **Principal Components**:
  - The first 10 principal components are visualized as 60x60 images.
  
- **Image Reconstruction**:
  - A comparison of the original image and its reconstructions using different numbers of principal components is shown side-by-side.

### Requirements
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

### Usage
1. Place grayscale dog images in the `q1_dog/` directory. Each image should be 60x60 pixels in size.
2. Run the provided script to perform PCA, visualize the principal components, and reconstruct the images using different numbers of principal components.
3. The program will display the first 10 principal components and the reconstructed images with varying `k` values.

### License
This project is licensed under the MIT License.
