"""
This script is used to compute the Gradient based feature from the image feature 
"""



import numpy as np
from filter import Filter

EPS = 1e-6



class GradientFeature:
    """
    Class to estimate the general gradient feature from the image feature
    """
    @staticmethod
    def create_circular_mask(radius: int) -> np.array:
        """
        Create a circular mask of radius radius.
        Inputs:
            - radius: radius of the mask in number of pixels.
        Outputs:
            - mask: 2*radius x 2*radius mask.
        """
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        return mask

    @staticmethod
    def generate_oriented_masks(radius: int, theta: float):
        """
        Generate left and right masks for a given orientation.
        Inputs:
            - radius: radius of the mask in number of pixels.
            - theta: orientation of the mask in radians.
        Outputs:
            - left_mask, right_mask: oriented masks.
        """
        mask = GradientFeature.create_circular_mask(radius)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        angles = np.arctan2(y, x)
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        relative_angles = np.arctan2(np.sin(angles - theta), np.cos(angles - theta))
        left_mask = np.logical_and(mask, relative_angles <= 0)
        right_mask = np.logical_and(mask, relative_angles > 0)
        return left_mask, right_mask

    @staticmethod
    def compute_histogram_gradient(img : np.array ,radius : int ,theta: float ,similarity_map :np.array = None):
        """
        Computes the histogram difference between the two halves of the mask 
        for image with n-discrete levels
        Inputs:
            - img : H x W  image with n-discrete levels 
            - radius : radius of the mask in number of pixels 
            - theta : orientation of the mask in radians
            - similarity_map : similarity  between the levels of the image 

        Outputs:
            - img_gradient : H x W gradient image
        """
        if img.ndim != 2:
            raise ValueError("Input image must be 2D.")
        if radius <= 0:
            raise ValueError("Radius must be positive.")

        left_mask, right_mask = GradientFeature.generate_oriented_masks(radius, theta)
        unique_values = np.unique(img)
        left_histograms, right_histograms = GradientFeature.compute_histograms(img, unique_values, left_mask, right_mask)

        # Vectorized computation of the gradient feature
        g_minus_h = left_histograms - right_histograms
        if similarity_map is not None:
            g_plus_h = left_histograms + right_histograms + EPS
            gradient_feature = 0.5 * np.nansum((g_minus_h ** 2) / g_plus_h, axis=2)
        else:
            gradient_feature = np.einsum('ijk,kl,ijl->ij', g_minus_h, similarity_map, g_minus_h)

        return gradient_feature

    @staticmethod
    def cgmo(img : np.array,radius : int ,n_orient : int ,similarity_map :np.array = None) -> np.array:
        """
        Compute compute_histogram_gradient feautre for multiple orientations which 
        are equally spaced between 0 and pi
        Inputs:
            - img : H x W  image with n-discrete levels 
            - radius : radius of the mask in number of pixels 
            - n_orient : number of orientations
            - similarity_map : similarity  between the levels of the image
        Outputs:
            - img_gradient : H x W x n_orient gradient image
        """
        if img.ndim != 2:
            raise ValueError("Input image must be 2D.")
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        if n_orient <= 0:
            raise ValueError("Number of orientations must be positive.")

        angles = np.linspace(0, np.pi, n_orient, endpoint=False)
        img_gradient = np.zeros((img.shape[0], img.shape[1], n_orient))
        for i, theta in enumerate(angles):
            img_gradient[:, :, i] = GradientFeature.compute_histogram_gradient(img, radius, theta, similarity_map)
        return img_gradient

    @staticmethod
    def compute_histograms(img, unique_values, left_mask, right_mask):
        """
        Computes histograms for left and right masks.
        """
        histograms_shape = (img.shape[0], img.shape[1], len(unique_values))
        left_histograms = np.zeros(histograms_shape)
        right_histograms = np.zeros(histograms_shape)

        for i, value in enumerate(unique_values):
            binary_img = (img == value).astype(int)
            left_histograms[:, :, i] = Filter.convolve2d(binary_img, left_mask, mode='same')
            right_histograms[:, :, i] = Filter.convolve2d(binary_img, right_mask, mode='same')
        return left_histograms, right_histograms

    @staticmethod
    def calculate_chi_squared(g, h, similarity_map):
        """
        Calculates the Chi-squared distance.
        """
        if similarity_map is None:
            return 0.5 * np.nansum(((g - h) ** 2) / (g + h + EPS))
        else:
            diff = g - h
            return np.dot(diff.T, np.dot(similarity_map, diff))
