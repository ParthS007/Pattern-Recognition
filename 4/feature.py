"""
This script contains functions which takes the image and generates the features from it
"""

from typing import Tuple, List
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from filter import Filter
from gradient_feature import GradientFeature



class FeatureExtractor:
    """
    Computes the texture, color and brightness gradient from the image
    """
    def __init__(self, n_bins=32,texton_features=None):
        self.n_bins = n_bins
        if texton_features is None:
            self.texton_features = pickle.load(open('./texton_feature_32.pkl', 'rb'))
        else:
            self.texton_features = texton_features


    def brigthness_gradient(self,img: np.array,radius : float,n_orient : int) -> np.array:
        """
        Computed the brightness gradients
        Inputs:
            - img : H x W x D image or H x W image if H x W x D image is passed,
              it is converted to grayscale by taking the mean of the channels
            - radius : radius of the mask  as the percentage of the image diagonal
            - n_orient : number of orientations
        Outputs:
            - brightness_grad : H x W x n_orient gradient image

        Note: 1. theta is in radians and sampled with equal spacing between 0 and pi
              2. Discritize the image into n_bins using np.digitize 
        """
        if img.ndim == 3:
            img = np.mean(img, axis=2)

        # Digitize the image into n_bins
        levels = np.linspace(np.min(img), np.max(img), self.n_bins)
        # to get bin indices starting from 0
        digitized_img = np.digitize(img, levels) - 1

        diag_length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
        pixel_radius = int(radius * diag_length)

        # Compute similarity map
        similarity_map = self.colorsim(pixel_radius)

        # Compute brightness gradient using cgmo
        brightness_grad = GradientFeature.cgmo(digitized_img, pixel_radius, n_orient, similarity_map)

        return brightness_grad

    def color_gradient(self, img : np.ndarray, radius : float ,n_orient : int ) -> np.array :
        """
        Compute the color gradient for the a color image
        Input: 
            - img :  H x W x D image if the size is H x W return the brigthness gradient
            - radius : radius of the mask  as the percentage of the image diagonal
            - n_orient : number of orientations
        Output:
            - color_grad : H x W x D x n_orient gradient image

        """
        # Check if the image is grayscale
        if img.ndim == 2 or img.shape[2] == 1:
            # The image is grayscale, compute brightness gradient
            color_grad = self.brigthness_gradient(img, radius, n_orient)
        else:
            # The image is color, compute brightness gradient for each channel
            channels = img.shape[2]
            H, W = img.shape[0], img.shape[1]
            color_grad = np.zeros((H, W, channels, n_orient))

            for c in range(channels):
                color_grad[:, :, c, :] = self.brigthness_gradient(img[:, :, c], radius, n_orient)

        return color_grad

    def texture_gradient(self, img : np.ndarray, radius : float ,n_orient : int) -> np.array:
        """
        Compute the texture gradient for a grayscale image using the texton features
        Input: 
            - img : H x W image
            - radius : radius of the mask  as the percentage of the image diagonal
            - n_orient : number of orientations
        Output:
            - texture_grad : H x W x n_orient gradient image
        """
        diag_length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
        pixel_radius = int(radius * diag_length)
        if img.ndim == 3:
            img = np.mean(img, axis=2)

        filterbanks = self.texton_features['fb']
        centroids = self.texton_features['tex']
        filters = self.extract_filters(filterbanks)

        filtered_output = self.filterbank_output(filters, img)
        # Ensure filtered_output is reshaped or transposed as needed
        # Example: flattening the spatial dimensions while preserving the feature dimension
        H, W, C = filtered_output.shape
        features = filtered_output.reshape(-1, C)  # Reshape to (H*W, C)

        # Compute distances and find the closest centroid for each feature
        distances = self.pairwise_distance(features, centroids)
        texture_feature_indices = np.argmin(distances, axis=1)

        # Reshape texture_feature_indices back to the spatial dimensions of the image
        texture_feature = texture_feature_indices.reshape(H, W)
        

        similarity_map = self.colorsim(pixel_radius)

        texture_grad = GradientFeature.cgmo(texture_feature, pixel_radius, n_orient, similarity_map)

        return texture_grad


    # This is optional, we will not evaluate it
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Combine the features to get a feature vectors
        1. Experiment with different combinations of features 
        Input:
            - img : H x W x 3 image
        Output:
            - feature_vector : H x W x n_features feature vector

        Example code:
            bg = self.brigthness_gradient(img,0.01,4)
            tg = self.texture_gradient(img,0.01,4)
            return np.concatenate((bg,tg),axis=2)

        """
        
        # COMPLETE THE CODE HERE
        return 

    def colorsim(self, sigma: float) -> np.ndarray:
        """
        Computes the color similarity map
        Input:
            - sigma: sigma of the Gaussian (You can use the radius of the mask as sigma)
        Output:
            - m: n_bins x n_bins color similarity map
        """
        bin_centers = np.linspace(0, 1, self.n_bins)
        bin_center_x, bin_center_y = np.meshgrid(bin_centers, bin_centers)
        m = 1 - np.exp(-((bin_center_x - bin_center_y)**2 / (2 * sigma**2)))
        return m

    @ staticmethod
    def extract_filters(filterbanks: dict) -> List[np.ndarray]:
        """
        Extracts the filters from the texton data and outputs a list of filters
        Input:
            - filterbanks: dictionary containing the filterbanks (from the pickle file)
        Output:
            - filter_list: list of filters
        Note: Use self.texton_features['fb'] to get the filterbanks
            and use this function to get the list of filters
        """
        fb1list = []
        fb2list =[]
        
        for f in filterbanks:
            fb1list.append(f[0])
            fb2list.append(f[1])
            
        return fb1list + fb2list
        
    @staticmethod
    def filterbank_output(filters_list: List[np.ndarray], image: np.ndarray) -> np.ndarray:
        """
        Computes the output of the filterbank for a particular image
        Input:
            -    filters_list: list of filters
            -    image: H x W image to be filtered 
        Output:
            -    image_filtered: H x W x n_filters filtered image
        Note: n_filters = len(filters_list)
        """
        image_filtered = np.zeros((image.shape[0],image.shape[1],len(filters_list)))
        for index, filt in enumerate(filters_list):
            image_filtered[:,:,index] = Filter.convolve2d_fft(image,filt,mode='same')     
        return image_filtered

    @staticmethod
    def pairwise_distance(features: np.array, centroids: np.array) -> np.array:
        """
        Computes the pairwise distance between the features and the centroids 
        Inputs: 
            features: (N x D) array of features
            centroids: (K x D) array of centroids
        output:
            distance: (N x K) array of pairwise distance between the features and the centroids
        """

        # If the feature dimensions of features and centroids don't match, pad the smaller one with zeros
        if features.shape[1] < centroids.shape[1]:
            features = np.pad(features, ((0, 0), (0, centroids.shape[1] - features.shape[1])))
        elif centroids.shape[1] < features.shape[1]:
            centroids = np.pad(centroids, ((0, 0), (0, features.shape[1] - centroids.shape[1])))

        distance =  -2 *(features @ centroids.T) + (
            np.linalg.norm(features,axis=1)**2)[:,None] + (np.linalg.norm(centroids,axis=1)**2)[None,:]
        return  np.sqrt(distance)
        









if __name__ == '__main__':

    # We use an image from the dataset 
    # Make sure to unzip the dataset in the same folder as this file

    img_sample = plt.imread('./BSDS300-images/BSDS300/images/train/24004.jpg')

    img_test = plt.imread('./BSDS300-images/BSDS300/images/train/22013.jpg')

    feature_extractor = FeatureExtractor(n_bins=32)
    
    print('Computing the brightness gradient')
    start = time.process_time()
    img_sample_bg = feature_extractor.brigthness_gradient(img_sample,0.01,8)
    print('Time taken for brightness gradient: ',time.process_time() - start)

    print('Computing the color gradient')
    start = time.process_time()
    img_sample_cg = feature_extractor.color_gradient(img_sample,0.01,8)
    print('Time taken for color gradient: ',time.process_time() - start)

    print('Computing the texture gradient')
    start = time.process_time()
    img_sample_tg = feature_extractor.texture_gradient(img_sample,0.01,8)
    print('Time taken for texture gradient: ',time.process_time() - start)

    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(img_sample)
    plt.subplot(1,4,2)
    plt.imshow(img_sample_bg[:,:,0])
    plt.subplot(1,4,3)
    plt.imshow(img_sample_cg[:,:,0,0])
    plt.subplot(1,4,4)
    plt.imshow(img_sample_tg[:,:,0])
    plt.show() 


