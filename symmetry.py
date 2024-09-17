# symmetry_detection.py

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from img2csv import image_to_csv


class SymmetryDetector:
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image at path {image_path} not found.")
        return img

    def split_image_vertical(self, image):
        height, width = image.shape
        mid = width // 2
        left_half = image[:, :mid]
        right_half = image[:, mid:]
        return left_half, right_half, mid

    def split_image_horizontal(self, image):
        height, width = image.shape
        mid = height // 2
        top_half = image[:mid, :]
        bottom_half = image[mid:, :]
        return top_half, bottom_half, mid

    def extract_diagonals(self, image):
        height, width = image.shape
        main_diagonal = np.array([image[i, i]
                                 for i in range(min(height, width))])
        anti_diagonal = np.array([image[i, width - i - 1]
                                 for i in range(min(height, width))])
        return main_diagonal, anti_diagonal

    def compare_images(self, img1, img2):
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        img1_cropped = img1[:min_height, :min_width]
        img2_cropped = img2[:min_height, :min_width]
        similarity_index, _ = ssim(img1_cropped, img2_cropped, full=True)
        return similarity_index

    def compare_diagonals(self, diag1, diag2):
        min_len = min(len(diag1), len(diag2))
        diag1_cropped = diag1[:min_len]
        diag2_cropped = diag2[:min_len]
        similarity_index = np.mean(diag1_cropped == diag2_cropped)
        return similarity_index

    def draw_symmetry_line(self, image, orientation, mid):
        if orientation == 'vertical':
            cv2.line(image, (mid, 0), (mid, image.shape[0]), (0, 255, 0), 2)
        elif orientation == 'horizontal':
            cv2.line(image, (0, mid), (image.shape[1], mid), (0, 255, 0), 2)
        elif orientation == 'main_diagonal':
            cv2.line(image, (0, 0),
                     (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        elif orientation == 'anti_diagonal':
            cv2.line(image, (image.shape[1], 0),
                     (0, image.shape[0]), (0, 255, 0), 2)
        return image

    def detect_and_draw_symmetry(self, image_path, output_path):
        image = self.load_image(image_path)

        left_half, right_half, mid_vertical = self.split_image_vertical(image)
        right_half_flipped = cv2.flip(right_half, 1)
        vertical_similarity = self.compare_images(
            left_half, right_half_flipped)

        top_half, bottom_half, mid_horizontal = self.split_image_horizontal(
            image)
        bottom_half_flipped = cv2.flip(bottom_half, 0)
        horizontal_similarity = self.compare_images(
            top_half, bottom_half_flipped)

        main_diagonal, anti_diagonal = self.extract_diagonals(image)
        main_diagonal_flipped = np.flipud(main_diagonal)
        anti_diagonal_flipped = np.flipud(anti_diagonal)

        main_diagonal_similarity = self.compare_diagonals(
            main_diagonal, main_diagonal_flipped)
        anti_diagonal_similarity = self.compare_diagonals(
            anti_diagonal, anti_diagonal_flipped)

        print(f"Vertical Structural Similarity Index: {vertical_similarity}")
        print(f"Horizontal Structural Similarity Index: { horizontal_similarity}")
        print(f"Main Diagonal Structural Similarity Index: {main_diagonal_similarity}")
        print(
            f"Anti-Diagonal Structural Similarity Index: {anti_diagonal_similarity}")

        # Load the color image to draw lines on it
        color_image = cv2.imread(image_path)

        # Draw symmetry lines if the similarity indices are above the threshold
        if vertical_similarity > self.threshold:
            color_image = self.draw_symmetry_line(
                color_image, 'vertical', mid_vertical)
        if horizontal_similarity > self.threshold:
            color_image = self.draw_symmetry_line(
                color_image, 'horizontal', mid_horizontal)
        if main_diagonal_similarity > self.threshold:
            color_image = self.draw_symmetry_line(
                color_image, 'main_diagonal', None)
        if anti_diagonal_similarity > self.threshold:
            color_image = self.draw_symmetry_line(
                color_image, 'anti_diagonal', None)

        # Save the image with or without symmetry lines
        cv2.imwrite(output_path, color_image)
        image_to_csv(output_path, output_path.replace('.png', '.csv'))

        # Return the symmetry status
        if any([vertical_similarity > self.threshold, horizontal_similarity > self.threshold,
                main_diagonal_similarity > self.threshold, anti_diagonal_similarity > self.threshold]):
            return "Symmetrical: Symmetry line(s) drawn"
        else:
            return "Asymmetrical"
