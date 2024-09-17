import cv2
import numpy as np
import pandas as pd

def image_to_csv(image_path, csv_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_array = np.array(image)
    df = pd.DataFrame(image_array)
    df.to_csv(csv_path, index=False, header=False)
