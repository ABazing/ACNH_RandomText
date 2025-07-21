# Filename: emote.py

import os
import re
import random
import sys

# --- CONFIGURATION ---
# The top-level folder to start searching from.
FOLDER_NAME = "b2_jock"

# The range for the random number (inclusive).
MIN_NUMBER = 0
MAX_NUMBER = 90
# ---------------------

def randomize_specific_tag(match_obj):
    """
    This function is called for every match found by the regex.
    It generates a new random number and reconstructs the tag,
    preserving the original whitespace.
    """
    original_whitespace = match_obj.group(2)
    random_number = random.randint(MIN_NUMBER, MAX_NUMBER)
    new_tag = f'{{{{40:{random_number}{original_whitespace}arg="0x00CD0000"}}}}'
    return new_tag

def process_files_in_folder(top_folder_path):
    """
    Main function to find all .txt files in a folder and its subfolders,
    and then process them.
    """
    print(f"--- Starting Tag Randomizer ---")
    
    if not os.path.isdir(top_folder_path):
        print(f"Error: Top-level folder '{top_folder_path}' not found.")
        os.makedirs(top_folder_path)
        print(f"Created folder '{top_folder_path}'. Please add your files/folders and run again.")
        return

    # --- THIS IS THE NEW PART ---
    # We will use os.walk to find ALL .txt files in all subdirectories.
    files_to_process = []
    print(f"Searching for .txt files in '{top_folder_path}' and all subfolders...")
    for dirpath, dirnames, filenames in os.walk(top_folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                # Construct the full, absolute path to the file and add it to our list.
                full_path = os.path.join(dirpath, filename)
                files_to_process.append(full_path)
    # --- END OF NEW PART ---

    if not files_to_process:
        print(f"No '.txt' files found anywhere inside '{top_folder_path}'.")
        return

    print(f"\nFound {len(files_to_process)} file(s) to process.")

    pattern = re.compile(r'\{\{40:(\d+)(\s+)arg="0x00CD0000"\}\}')

    # Now we loop through the full paths we found.
    for file_path in files_to_process:
        # We print the full path so you know exactly which file is being processed.
        print(f"\nProcessing '{file_path}'...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            modified_content, num_replacements = pattern.subn(randomize_specific_tag, content)

            if num_replacements > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                print(f"Success! Found and randomized {num_replacements} tag(s).")
            else:
                print(f"No matching tags found. File was not changed.")

        except FileNotFoundError:
            print(f"Error: Could not find file '{file_path}' during processing loop.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file_path}': {e}")
            
    print("\n--- Script finished ---")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    target_folder = os.path.join(script_dir, FOLDER_NAME)
    
    process_files_in_folder(target_folder)
    input("\nPress Enter to exit...")