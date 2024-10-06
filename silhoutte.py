import cv2
import numpy as np

# Input image and coordinates
image = cv2.imread("path_to_image.jpg")  # Replace with your image path
coordinates = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # Replace with your four coordinates

# Create an empty 64x64 black image (silhouette)
silhouette = np.zeros((64, 64), dtype=np.uint8)

# Resize the human's bounding box region to 64x64
x_coords = [coord[0] for coord in coordinates]
y_coords = [coord[1] for coord in coordinates]

# Extract bounding box of the human in the original image
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# Crop the region containing the human
human_region = image[y_min:y_max, x_min:x_max]

# Resize the cropped region to 64x64
human_region_resized = cv2.resize(human_region, (64, 64))

# Create a mask using the polygon with the coordinates
polygon = np.array([[(int((x - x_min) / (x_max - x_min) * 64), int((y - y_min) / (y_max - y_min) * 64)) for x, y in coordinates]], dtype=np.int32)

# Fill the polygon (human shape) with white (255)
cv2.fillPoly(silhouette, polygon, 255)

# Save or show the silhouette
cv2.imwrite("human_silhouette.png", silhouette)
cv2.imshow("Silhouette", silhouette)
cv2.waitKey(0)
cv2.destroyAllWindows()
