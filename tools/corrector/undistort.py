import cv2
import numpy as np
import json
import argparse
import os

def undistort_image(image_path, calibration_file, output_path=None, show=False):
    """
    Undistort an image using camera calibration parameters.
    """
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file '{calibration_file}' not found.")
        return

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coeffs'])
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    h, w = img.shape[:2]
    
    # Refine camera matrix (optional, helps preserve valid pixels)
    # alpha=1: all pixels are retained, black borders may appear
    # alpha=0: crop to valid pixels only
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image (optional, based on ROI)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    
    if output_path:
        cv2.imwrite(output_path, dst)
        print(f"Undistorted image saved to: {output_path}")

    if show:
        # Resize for display if too large
        display_scale = 0.5
        small_src = cv2.resize(img, (0,0), fx=display_scale, fy=display_scale)
        small_dst = cv2.resize(dst, (0,0), fx=display_scale, fy=display_scale)
        
        combined = np.hstack((small_src, small_dst))
        cv2.imshow('Original vs Undistorted', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def print_camera_specs(calibration_file):
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file '{calibration_file}' not found.")
        return

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coeffs'])
    width = data.get('image_width')
    height = data.get('image_height')
    error = data.get('reprojection_error')

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("--- Camera Calibration Results ---")
    print(f"Resolution: {width} x {height}")
    print(f"Reprojection Error: {error:.4f} (lower is better)")
    print(f"Focal Length (fx, fy): ({fx:.2f}, {fy:.2f}) pixels")
    print(f"Principal Point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels")
    
    k1, k2, p1, p2, k3 = dist[0]
    print(f"Distortion Coefficients (k1, k2, p1, p2, k3):")
    print(f"  Radial: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}")
    print(f"  Tangential: p1={p1:.4f}, p2={p2:.4f}")
    
    # Calculate FOV (Field of View)
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    print(f"Field of View (Approx):")
    print(f"  Horizontal FOV: {fov_x:.2f} degrees")
    print(f"  Vertical FOV:   {fov_y:.2f} degrees")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort images using camera calibration parameters.")
    parser.add_argument("--image", type=str, help="Path to image to undistort")
    parser.add_argument("--calib", type=str, default="calibration_result.json", help="Path to calibration JSON file")
    parser.add_argument("--output", type=str, default="undistorted.jpg", help="Output path for undistorted image")
    parser.add_argument("--info", action="store_true", help="Print camera specs only")
    parser.add_argument("--show", action="store_true", help="Show comparison window")

    args = parser.parse_args()

    if args.info:
        print_camera_specs(args.calib)
    elif args.image:
        undistort_image(args.image, args.calib, args.output, args.show)
    else:
        # Default behavior: print info if no image provided
        print_camera_specs(args.calib)
