import cv2
import numpy as np
from img2csv import image_to_csv

def detect_shapes(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(image)
    def is_star(contour, approx):
        if len(approx) == 10:
            return True
        return False
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 4 and hierarchy[0][i][3] == -1:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                print("square")
                cv2.rectangle(output, (x, y), (x + w, y + h),(255, 255, 255), 2)
            else:
                print("rectangle")
                cv2.rectangle(output, (x, y), (x + w, y + h),(255, 255, 255), 2)
        elif is_star(contour, approx):
            print("star")
            cv2.drawContours(output, [contour], -1, (255, 255, 255), 2)
        elif len(approx) > 4:
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * (area / (peri * peri))
            if 0.7 < circularity <= 1.2:
                print("circle")
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output, center, radius, (255, 255, 255), 2)
        elif len(approx) > 4:
            if len(contour) >= 5:
                print("ellipse")
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(output, ellipse, (255, 255, 255), 2)
    cv2.imwrite(output_path, output)
    image_to_csv(output_path, output_path.replace('.png', '.csv'))
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
