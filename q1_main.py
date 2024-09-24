import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def main():
    directory = "./q1_dog/"
    if not os.path.isdir(directory):
        print("Directory not found.")
        exit()

    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

    X = np.zeros((len(images), 3600))

    for i in range(len(images)):
        img = images[i]
        flattened_image = img.flatten()
        for j in range(len(flattened_image)):
            X[i, j] = flattened_image[j]



    mean_vec = np.mean(X, axis=0)
    var_vec = np.std(X,axis=0)
    centered_X = X - mean_vec
    centered_X = centered_X/var_vec #normalization
    cov_matrix = np.cov(centered_X, rowvar=False)#cov matrix function use

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    top_k = 10
    top_eigenvectors = sorted_eigenvectors[:, :top_k]
    total_variance = np.sum(sorted_eigenvalues)
    pve = sorted_eigenvalues / total_variance


    fig, axes = plt.subplots(2, 5)
    axes = axes.flatten()

    for i in range(10):
        pc = top_eigenvectors[:, i]
        pc_image = np.reshape(pc, (60, 60))
        axes[i].imshow(pc_image, cmap='gray')
        axes[i].set_title(f"Principal Component {i+1}")

    plt.tight_layout()
    plt.show()




    def reconstruct_image(image, eigenvectors, k):
        original_shape = (60,60) #required shape
        #image = image.reshape(original_shape)
        selected_eigenvectors = eigenvectors[:, :k]
        coefficients = np.dot(image.T, selected_eigenvectors)
        reconstructed_image = np.dot(coefficients, selected_eigenvectors.T)
        reconstructed_image = reconstructed_image.reshape(original_shape)
        return reconstructed_image

    k_values = [1, 5, 10, 20, 50, 500,3600]

    plt.figure(figsize=(10, 3))
    plt.subplot(1, len(k_values) + 1, 1)
    plt.imshow(images[0], cmap='gray')
    plt.title('Original')

    for i in range(len(k_values)):
        k = k_values[i]
        reconstructed = reconstruct_image(X[0,:], sorted_eigenvectors, k)
        plt.subplot(1, len(k_values) + 1, i + 2)
        plt.imshow(reconstructed, cmap='gray')
        plt.title(f'k={k}')


    plt.tight_layout()
    plt.show()
    return

main()