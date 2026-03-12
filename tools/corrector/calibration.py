import cv2
import numpy as np
import glob
import os
import argparse
import json

def calibrate_camera(image_dir, rows, cols, output_file, square_size=2.0, show_images=False):
    """
    Calibrate camera using chessboard images.
    
    Args:
        image_dir (str): Directory containing calibration images.
        rows (int): Number of inner corners per row.
        cols (int): Number of inner corners per column.
        output_file (str): Path to save calibration results (JSON).
        square_size (float): Size of a square in real-world units (e.g., mm or m).
        show_images (bool): Whether to show images with detected corners.
    """
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}. Starting calibration...")

    image_size = None
    valid_images_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]
        elif image_size != gray.shape[::-1]:
            print(f"Warning: Image {fname} has different size {gray.shape[::-1]} than first image {image_size}. Skipping.")
            continue

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            valid_images_count += 1
            print(f"Corners found in {os.path.basename(fname)}")

            # Draw and display the corners
            if show_images:
                cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        else:
            print(f"Corners NOT found in {os.path.basename(fname)}")

    if show_images:
        cv2.destroyAllWindows()

    if valid_images_count < 1:
        print("Not enough valid images for calibration.")
        return

    print(f"Calibrating with {valid_images_count} valid images...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if not ret:
        print("Calibration failed.")
        return

    print("Calibration successful!")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")

    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    total_error = mean_error / len(objpoints)
    print(f"Total Re-projection Error: {total_error}")

    # Save results
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "reprojection_error": total_error
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        print(f"Calibration results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing calibration images (default: 'images' subdirectory or current directory)")
    parser.add_argument("--rows", type=int, default=10, help="Number of inner corners per row (height)")
    parser.add_argument("--cols", type=int, default=7, help="Number of inner corners per column (width)")
    parser.add_argument("--square_size", type=float, default=2.0, help="Size of a square in real-world units")
    parser.add_argument("--output", type=str, default="tools\corrector\calibration_result.json", help="Output JSON file for calibration results")
    parser.add_argument("--show", action="store_true", help="Show images with detected corners")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine image directory
    if args.image_dir is None:
        # Try 'images' subdirectory first
        possible_dir = os.path.join(script_dir, "images")
        if os.path.exists(possible_dir):
            image_dir = possible_dir
        else:
            # Fallback to script directory
            image_dir = script_dir
            print(f"Default 'images' directory not found. Scanning script directory: {image_dir}")
    else:
        # Resolve user-provided path
        if not os.path.isabs(args.image_dir):
            image_dir = os.path.join(script_dir, args.image_dir)
        else:
            image_dir = args.image_dir

    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.")
        exit(1)

    calibrate_camera(image_dir, args.rows, args.cols, args.output, args.square_size, args.show)
