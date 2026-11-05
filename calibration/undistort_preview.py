
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

PARAM_FILE = BASE_DIR / "calibration" / "camera_params.npz"
IMAGE_FILE = BASE_DIR / "calibration" / "images" / "checkerboard_01.jpg"
OUTPUT_FILE = BASE_DIR / "calibration" / "undistorted_preview.jpg"

# Load saved camera parameters
data = np.load(PARAM_FILE)

camera_matrix = data["camera_matrix"]
distortion = data["distortion_coefficients"]

# Load original image
image = cv2.imread(str(IMAGE_FILE))

if image is None:
    raise SystemExit("Image not found!")

height, width = image.shape[:2]

# Check image resolution
if (width, height) != (
    int(data["image_width"]),
    int(data["image_height"])
):
    raise SystemExit("Image resolution does not match calibration!")

# Calculate optimized camera matrix
new_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    distortion,
    (width, height),
    1,
    (width, height)
)

# Correct lens distortion
undistorted = cv2.undistort(
    image,
    camera_matrix,
    distortion,
    None,
    new_matrix
)

# Save corrected image
cv2.imwrite(str(OUTPUT_FILE), undistorted)

print("Original image:", IMAGE_FILE)
print("Corrected image:", OUTPUT_FILE)
print("Undistortion completed successfully!")
