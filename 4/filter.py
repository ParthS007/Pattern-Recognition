from typing import Tuple
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt



class Filter():
    """
     Class containing the basic filtering functions
    """
    @staticmethod
    def convolve2d(image: np.ndarray, image_filter: np.ndarray, mode: str ='same') -> np.ndarray:
        """
        To compute the convolution of the image with the filter
        Input:
            image : image of the form K x K, where KxK is the dimension of the image
            imageFilter : filter of the form k x k, where kxk is the dimension of the filter
            mode : mode of the convolution, either 'valid' or 'same'
        Output:
            convolvedImage : convolved image of the form K x K, where KxK is the dimension of the image if mode is 'same' 
            and (K-k+1) x (K-k+1) if mode is 'valid'

        Convolve the image using zero padding and the for loops. For the border mode same, we need to pad the image with zeros.
        This is to ensure that the filter is applied to all the pixels of the image.
        Hint: 
            1. You can use the function np.pad to pad the image with zeros
            2. You can use the function np.flip to flip the filter
            3. The convolution operator requires the filter to be flipped, here.
        Note: In general the filters need not be flipped, but here we need to flip the filter
               to get the correct output and follow the definition of convolution
        """

        # get dimensions of image and filter
        m, n = image.shape
        p, q = image_filter.shape


        # pad image with zeros
        image_padded = np.zeros((m + p - 1, n + q - 1))
        image_padded[p//2:m+p//2, q//2:n+q//2] = image

        # initialize output array
        output = np.zeros((m, n))
        #Flip the kernel
        image_filter = np.flip(image_filter)
        # loop over every pixel of the output
        for i in range(m):
            for j in range(n):
                # element-wise multiplication of the kernel and the image
                output[i, j] = ( image_filter * image_padded[i:i+p, j:j+q]).sum()

        if(mode == 'same'):
            return output
        elif(mode == 'valid'):
            return output[p//2:m-p//2, q//2:n-q//2]
        else:
            raise ValueError('mode must be either valid or same')

    @staticmethod
    def convolve2d_fft(image: np.ndarray, image_filter: np.ndarray, mode: str ='same')-> np.ndarray:      
        """
        To compute the convolution of the image with the filter using fft
        Input:
            image : image of the form K x K, where KxK is the dimension of the image
            imageFilter : filter of the form k x k, where kxk is the dimension of the filter
            mode : mode of the convolution, either 'valid' or 'same' 
        Output:
            convolvedImage : convolved image of the form K x K, where KxK is the dimension of the image if mode is 'same'
                                 and (K-k+1) x (K-k+1) if mode is 'valid'
        Convolve the image using fft. For the border mode same, we need to pad the image with zeros. This
        is to ensure the fourier based convolution is same as the for loop based convolution in the boundary regions.
        Hint:
            1. You can use the function np.pad to pad the image with zeros
            2. Fliping the filter is not required here as multiplication in frequency 
                domain implies convolution in spatial domain
            3. Make sure to return the real part of the output
        """


        # get dimensions of image and kernel
        m, n = image.shape
        p, q = image_filter.shape

        # pad image with zeros
        image_padded = np.zeros((m + p - 1, n + q - 1))
        image_padded[p//2:m+p//2, q//2:n+q//2] = image

        # compute DFT of image and kernel
        F = np.fft.fft2(image_padded)
        G = np.fft.fft2(image_filter, s=(m + p - 1, n + q - 1))

        # element-wise multiplication of the transformed images
        H = F * G

        # compute inverse DFT
        h = np.fft.ifft2(H)

        # crop output to correct size
        if(mode == 'same'):
            h = h[p-1:m+p-1, q-1:n+q-1]
        elif(mode == 'valid'):
            h = h[p+ p//2-1: m + p//2, q+ q//2-1: n + q//2]
        else:
            raise ValueError('mode must be either valid or same')

        return np.real(h)
    

    @staticmethod
    def gaussian_lowpass( eta: float) -> np.ndarray:
        """
        To generate the Gaussian Low Pass Filter
        Input:
            sigma : standard deviation of the gaussian distribution
        Output:
            gaussianLowPassFilter : gaussian low pass filter of the form (2m+1) x (2m+1), where (2m+1) x (2m+1) is the dimension of the filter and 
                                    m = 4*eta. IF m is not an integer, then round it to smallest integer greater than m.
        Generate the gaussian low pass filter of the given shape and sigma. 
        Note:
            1.The filter should be normalized such that the sum of all the elements is 1.
            2.The pdf of the Gaussian is sampled at interger values between -m to m, where m is the size of the filter.
            3. We choose odd sized filters to ensure that the filter is symmetric about the center pixel.
        Hint:
            1. You can use the function np.meshgrid to generate the grid and samples along the grid.

        """
        #Define shape based on signma
        if(eta == 0):
            return np.array([[1]])
        m = int(np.ceil(4*eta)) # Fix the shape of the filter based on the sigm
        x,y = np.meshgrid(np.linspace(-m,m,2*m+1),np.linspace(-m,m,2*m+1))
        d = np.sqrt(x*x+y*y)
        g = np.exp(-( (d)**2 / ( 2.0 * eta**2 ) ) )
        g = g/np.sum(g)
        return g

    @staticmethod
    def gaussian_highpass(eta: float) -> np.ndarray:
        """
        To generate the Gaussian High Pass Filter
        Input:
            sigma : standard deviation of the gaussian distribution
        Output:
            gaussianHighPassFilter : gaussian high pass filter of the form (2m+1) x (2m+1),  where (2m+1) x (2m+1) is the dimension of the filter and
                                       m = 4*eta. IF m is not an integer, then round it to smallest integer greater than m.
        Hint:
            1.Use the relation gaussianHighPassFilter = delta - gaussianLowPassFilter.
            2. Delta is filter with only the center element as 1 and rest as 0.
        """
        lpf = Filter.gaussian_lowpass(eta)
        delta_function = np.zeros_like(lpf)
        m  = lpf.shape[0]//2
        delta_function[m,m] = 1
        return delta_function-lpf


    


if __name__ == '__main__':

    # Note: AutoGrader will not run this section of the code
    # You can use this to test your code and import any libraries here
    # Autograder doesnt have skimage or scipy installed
    from skimage.data import brick
    from scipy.signal import convolve2d,fftconvolve
    image = brick()

    # plt.imshow(image)
    # plt.show()

    # Test convolve2d
    image_filter = np.array([[1,2,3],[4,5,6],[7,8,9]])
    convolved_image_same = Filter.convolve2d(image,image_filter,mode='same')
    convolved_image_valid = Filter.convolve2d(image,image_filter,mode='valid')

    

    convolved_image_scipy_same = convolve2d(image,image_filter,mode='same')
    convolved_image_scipy_valid = convolve2d(image,image_filter,mode='valid')

    # Difference between your convolve2d and scipy convolve2d should be very small ideally zero
    difference = np.sum(np.abs(convolved_image_same-convolved_image_scipy_same))
    print('Convole2D (same) difference: ',difference)

    difference = np.sum(np.abs(convolved_image_valid-convolved_image_scipy_valid))
    print('Convole2D (valid) difference: ',difference)

    # Test convolve2d_fft
    convolved_image_fft_same = Filter.convolve2d_fft(image,image_filter,mode='same')
    convolved_image_fft_scipy_same = fftconvolve(image,image_filter,mode='same')

    convolved_image_fft_valid = Filter.convolve2d_fft(image,image_filter,mode='valid')
    convolved_image_fft_scipy_valid = fftconvolve(image,image_filter,mode='valid')

    # Difference between your convolve2d_fft and scipy convolve2d_fft should be very small ideally zero
    difference = np.sum(np.abs(convolved_image_fft_same-convolved_image_fft_scipy_same))
    print('Convole2D FFT (same) difference: ',difference)

    difference = np.sum(np.abs(convolved_image_fft_valid-convolved_image_fft_scipy_valid))
    print('Convole2D FFT (valid) difference: ',difference)


    # Test gaussian_lowpass
    eta_Val = 0.2
    gaussian_lowpass_filter = Filter.gaussian_lowpass(eta_Val)
    # Note this is for the case when eta = 0.2
    gaussian_expected = np.array([[1.38877368e-11, 3.72659762e-06, 1.38877368e-11],
       [3.72659762e-06, 9.99985094e-01, 3.72659762e-06],
       [1.38877368e-11, 3.72659762e-06, 1.38877368e-11]])
    

    
    if np.all(np.isclose(gaussian_lowpass_filter, gaussian_expected)):
        print("Passed the gaussian lowpass filter test")

    # Check if it sums to 1
    if np.isclose(np.sum(gaussian_lowpass_filter),1):
        print("Passed the gaussian lowpass filter normalization test")

    # Test gaussian_highpass
    gaussian_highpass_filter = Filter.gaussian_highpass(eta_Val)

    # Note this is for the case when eta = 0.2
    highpass_expected = np.array([[-1.38877368e-11, -3.72659762e-06, -1.38877368e-11],
       [-3.72659762e-06,  1.49064460e-05, -3.72659762e-06],
       [-1.38877368e-11, -3.72659762e-06, -1.38877368e-11]])
    
    if np.all(np.isclose(gaussian_highpass_filter, highpass_expected)):
        print("Passed the gaussian highpass filter test")

    # Check if it sums to 0
    if np.isclose(np.sum(gaussian_highpass_filter),0):
        print("Passed the gaussian highpass filter normalization test")


 