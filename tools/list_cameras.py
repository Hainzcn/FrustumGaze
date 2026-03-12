import subprocess
import json
import cv2

def get_camera_info_windows():
    # PowerShell command to get camera devices
    # Note: We filter by Class 'Camera' and 'Image' (for older webcams)
    cmd = [
        "powershell",
        "-Command",
        "Get-PnpDevice | Where-Object { $_.Class -eq 'Camera' -or $_.Class -eq 'Image' } | Select-Object FriendlyName, InstanceId, Status | ConvertTo-Json"
    ]
    try:
        # Use a specific encoding if necessary, but usually defaults work for English systems
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"PowerShell Error: {result.stderr}")
            return []
            
        output = result.stdout.strip()
        if not output:
            return []
            
        devices = json.loads(output)
        if isinstance(devices, dict):
            devices = [devices]
            
        # Filter only devices with status 'OK' if possible, but 'Status' might be returned as string
        valid_devices = [d for d in devices if d.get('Status') == 'OK']
        return valid_devices
    except Exception as e:
        print(f"Error getting camera info: {e}")
        return []

def list_cameras():
    print("--- System Camera Devices (PowerShell) ---")
    devices = get_camera_info_windows()
    for i, dev in enumerate(devices):
        print(f"Device {i}:")
        print(f"  Name: {dev.get('FriendlyName')}")
        print(f"  ID:   {dev.get('InstanceId')}")
        print(f"  Status: {dev.get('Status')}")

    print("\n--- OpenCV Camera Indices ---")
    for i in range(5):
        # Try DSHOW first as it maps better to Windows devices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        backend = "DSHOW"
        if not cap.isOpened():
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            backend = "ANY"
        
        if cap.isOpened():
            print(f"Index {i}: Opened ({backend})")
            # Try to get some property to identify? OpenCV doesn't give much
            cap.release()

if __name__ == "__main__":
    list_cameras()
