from IPython.display import clear_output
import os

def is_image_file(filename):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return os.path.splitext(filename)[1].lower() in image_extensions

few_image_dirs = {}
scan_count = 0  # counter

for root, dirs, files in os.walk('/kaggle/input'):
    scan_count += 1
    if scan_count % 10 == 0:
        clear_output(wait=True)
        print(f"Scanning... checked {scan_count} folders so far")

    image_files = [f for f in files if is_image_file(f)]
    
    if 0 < len(image_files) <= 2:
        dir_id = os.path.basename(root)
        few_image_dirs[dir_id] = image_files

# Final output
clear_output(wait=True)
print("Scan complete.\n")

if few_image_dirs:
    print("Directories with 2 or fewer image files:\n")
    for dir_name, images in few_image_dirs.items():
        print(f"Directory: {dir_name}")
        print("Image files:", images)
        print("-" * 40)
else:
    print("No directories with 2 or fewer image files found.")
