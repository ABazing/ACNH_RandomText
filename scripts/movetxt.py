import os
import shutil

def move_txt_files(source_root, target_folder):
    """
    Move all .txt files from source_root and its subfolders to target_folder.

    Args:
        source_root (str): The root folder to search for .txt files (e.g., 'C:/path/to/modded').
        target_folder (str): The folder where all .txt files will be moved (e.g., 'C:/path/to/txt_files').
    """
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Counter for moved files
    moved_count = 0

    # Walk through source_root and its subfolders
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith(".txt"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, file)

                # Check if file already exists in target; if so, append a number to avoid overwriting
                base_name, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(target_folder, f"{base_name}_{counter}{ext}")
                    counter += 1

                # Move the file
                shutil.move(source_path, target_path)
                moved_count += 1
                print(f"Moved: {source_path} -> {target_path}")

    print(f"Total .txt files moved: {moved_count}")

# Get user input
source_root = input("Enter the source folder path (e.g., C:/path/to/modded): ").strip()
target_folder = input("Enter the target folder path (e.g., C:/path/to/txt_files): ").strip()

# Run the function
move_txt_files(source_root, target_folder)