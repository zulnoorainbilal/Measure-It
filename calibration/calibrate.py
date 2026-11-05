
from pathlib import Path
import cv2
import numpy as np

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = BASE_DIR / "calibration" / "images"

OUTPUT_FILE = BASE_DIR / "calibration" / "camera_params.npz"

# Number of INNER corners (columns, rows)
CHECKERBOARD = (9, 7)

# Physical size of one square in millimeters
SQUARE_SIZE_MM = 20.0

MIN_IMAGES = 20


# ==========================================
# LOAD CALIBRATION IMAGES
# ==========================================

image_paths = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(IMAGE_DIR.glob(extension))

image_paths = sorted(image_paths)

print("Total images found:", len(image_paths))

if len(image_paths) < MIN_IMAGES:
    raise SystemExit(
        f"ERROR: At least {MIN_IMAGES} calibration images are required."
    )


# ==========================================
# PREPARE 3D OBJECT POINTS
# ==========================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE_MM


# ==========================================
# STORE DETECTED CORNERS
# ==========================================

object_points = []
image_points = []

image_size = None

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ==========================================
# PROCESS CHECKERBOARD IMAGES
# ==========================================

for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    current_size = (gray.shape[1], gray.shape[0])

    print(image_path.name, "Resolution:", current_size)

    if image_size is None:
        image_size = current_size

    elif current_size != image_size:
        print("Skipping different resolution:", image_path.name)
        continue

    found = False
    refined_corners = None

    # Try modern checkerboard detector
    if hasattr(cv2, "findChessboardCornersSB"):

        result = cv2.findChessboardCornersSB(
            gray,
            CHECKERBOARD,
            flags=(
    cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_EXHAUSTIVE
    + cv2.CALIB_CB_ACCURACY
)
        )

        # OpenCV returns detection status and corner coordinates
        if isinstance(result, tuple):

            found, corners = result

        else:

            corners = result
            found = corners is not None

        if found and corners is not None:

            refined_corners = np.asarray(
                corners,
                dtype=np.float32
            ).reshape(-1, 1, 2)

    # Fallback to standard detector
    else:

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        )

        if found and corners is not None:

            refined_corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            refined_corners = np.asarray(
                refined_corners,
                dtype=np.float32
            ).reshape(-1, 1, 2)

    # Save successful detections
    if found and refined_corners is not None:

        object_points.append(
            np.asarray(objp.copy(), dtype=np.float32).reshape(-1, 3)
        )

        image_points.append(
            np.asarray(refined_corners, dtype=np.float32).reshape(-1, 1, 2)
        )

        print("Detected:", image_path.name)

    else:

        print("Checkerboard not detected:", image_path.name)


# ==========================================
# CHECK SUCCESSFUL DETECTIONS
# ==========================================

successful_count = len(object_points)

print("\nSuccessful images:", successful_count)

if successful_count < MIN_IMAGES:

    raise SystemExit(
        f"ERROR: Need at least {MIN_IMAGES} successful checkerboard detections. "
        f"Only {successful_count} detected."
    )


# ==========================================
# CAMERA CALIBRATION
# ==========================================

rms_error, camera_matrix, distortion, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)


# ==========================================
# CALCULATE REPROJECTION ERROR
# ==========================================

errors = []

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion
    )

    error = cv2.norm(
        image_points[i],
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    errors.append(error)

mean_error = float(np.mean(errors))


# ==========================================
# SAVE CALIBRATION PARAMETERS
# ==========================================

np.savez_compressed(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion,
    image_width=image_size[0],
    image_height=image_size[1],
    rms_error=rms_error,
    mean_reprojection_error=mean_error
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n========== CALIBRATION RESULTS ==========")

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(distortion)

print("\nRMS Calibration Error:", rms_error)

print("Mean Reprojection Error:", mean_error, "pixels")

print("\nCalibration file saved at:")
print(OUTPUT_FILE)

print("\nCAMERA CALIBRATION COMPLETED!")
