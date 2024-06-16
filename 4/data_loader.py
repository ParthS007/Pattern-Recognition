"""
This script loads the image and the corresponding segementation map
"""
import os
from typing import Tuple, List
import numpy as np

import matplotlib.pyplot as plt


class BSDS300Loader:
    """
    Class to load the image and the corresponding segmentation map
    """
    def __init__(self, image_path = './BSDS300-images/BSDS300/images/train/', seg_path = './BSDS300-human/BSDS300/human/color/'):
        self.image_path = image_path
        self.seg_path = seg_path
        self.image_list = self.load_list()


    def load_list(self):
        """
        load the list of image names in the folder and save them as list of integers
        """
        image_list = os.listdir(self.image_path)
        image_list = [int(name.split('.')[0]) for name in image_list]
        image_list.sort()
        return image_list

    def  load_data(self,img_int : int )-> Tuple[ np.ndarray, np.ndarray]:
        """
        Given the integer value of the image, load the image and 
        the corresponding segmentation map. The segmentation map is the union of 
        all the human annotations.
        Inputs:
            - img_int : integer value of the image
        Outputs:
            - img : H x W x D image 
            - img_seg : H x W binary segmentation map
        Note: 
            - 1 for boundary and 0 for non-boundary
        """
        img_file = os.path.join(self.image_path, f"{img_int}.jpg")
        if not os.path.isfile(img_file):
            raise FileNotFoundError(f"Image file '{img_file}' not found.")
        img = plt.imread(img_file)
        img_seg = self.load_seg(img_int, img.shape[:2])
        return img, img_seg

    def load_seg(self, img_int : int , img_shape : Tuple) -> np.ndarray:
        """
        Load the segmentation map for the image from all the human annotations
        Inputs:
            - img_int : integer value of the image
            - img_shape : shape of the image 
        Outputs:
            - img_seg : H x W binary segmentation map
        """
        if not isinstance(img_shape, tuple) or len(img_shape) != 2:
            raise ValueError(
                "img_shape must be a tuple of two elements (height, width)."
            )
        # Initialize a binary segmentation map
        img_seg = np.zeros(img_shape, dtype=np.uint8)
        # Construct the list of segmentation files for this image
        seg_files = [
            os.path.join(self.seg_path, human_subj, f"{img_int}.seg")
            for human_subj in os.listdir(self.seg_path)
            if os.path.exists(os.path.join(self.seg_path, human_subj, f"{img_int}.seg"))
        ]
        # Load the segmentation map from each file and combine them
        for seg_file in seg_files:
            temp_seg_map = np.zeros(img_shape, dtype=np.uint8)
            with open(seg_file, "r") as file:
                lines = file.readlines()
                data_lines = lines[12:]
                for line in data_lines:

                    line_vars = line.strip().split()
                    s, row, c1, c2 = [int(var) for var in line_vars]
                    if (
                        s != 0
                        and row < img_shape[0]
                        and c1 < img_shape[1]
                        and c2 < img_shape[1]
                    ):
                        temp_seg_map[row, c1 : c2 + 1] = 1
            # Combine the individual segmentation with the overall map
            img_seg = np.logical_or(img_seg, temp_seg_map)
        print(f"Total boundary pixels in image {img_int}: {np.sum(img_seg)}")
        return img_seg

if __name__ == '__main__':

    # Make sure to unzip the dataset in the same folder as this file
    # Note here well be using the segmentation from one of the annotations
    REFERENCE_SEG_PATH  = './sample_segmentation.npy'

    TRAIN_PATH = './BSDS300-images/BSDS300/images/train/'
    SEG_PATH = './BSDS300-human/BSDS300/human/color/'
    train_loader = BSDS300Loader(TRAIN_PATH,SEG_PATH)



    img_sample_seg = np.load(REFERENCE_SEG_PATH)

    SAMPLE_INDEX = 198023
    img_sample,seg_map_sample = train_loader.load_data(SAMPLE_INDEX)

    print("Differne between the loaded segmentation and the reference segmentation: ",
     np.linalg.norm(img_sample_seg - seg_map_sample))

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_sample)
    plt.subplot(1,2,2)
    plt.imshow(seg_map_sample)
    plt.show()