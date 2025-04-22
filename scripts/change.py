import os
import re
import random
import sys
import logging
from nltk.corpus import wordnet
from tqdm import tqdm

# Ensure NLTK WordNet is downloaded (run once manually if needed: nltk.download('wordnet'))
import nltk
nltk.data.path.append(r"C:\nltk_data")  # Adjust if needed for your system

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("synonym_antonym_changes.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_synonym_or_antonym(word):
    """Get a random synonym or antonym for a word with variety, or return original if none found."""
    # 50/50 chance for synonym or antonym if replacement is triggered
    choice = random.choice(["synonym", "antonym"])

    synsets = wordnet.synsets(word)
    if not synsets:
        return word  # No synsets, keep original

    # Get all synonyms
    synonyms = []
    for syn in synsets:
        for lemma in syn.lemmas():
            syn_word = lemma.name().replace("_", " ")
            if syn_word != word and syn_word not in synonyms:  # Avoid duplicates and original
                synonyms.append(syn_word)
    random.shuffle(synonyms)  # Shuffle for variety

    # Get all antonyms
    antonyms = []
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.antonyms():
                ant_word = lemma.antonyms()[0].name().replace("_", " ")
                if ant_word != word and ant_word not in antonyms:  # Avoid duplicates and original
                    antonyms.append(ant_word)
    random.shuffle(antonyms)  # Shuffle for variety

    if choice == "synonym" and synonyms:
        new_word = synonyms[0]
        logger.info(f"Changed '{word}' to synonym '{new_word}'")
        return new_word
    elif choice == "antonym" and antonyms:
        new_word = antonyms[0]
        logger.info(f"Changed '{word}' to antonym '{new_word}'")
        return new_word
    return word  # Default to original if no replacement found

def mangle_text(text):
    """Replace random words (3+ letters) with synonyms or antonyms outside tags with 30% chance."""
    if not text.strip():
        return text

    # Split around tags
    parts = re.split(r"(\{\{.*?\}\})", text)
    new_parts = []

    for part in parts:
        if part.startswith("{{") and part.endswith("}}"):
            new_parts.append(part)  # Keep tags unchanged
        else:
            words = part.split()
            if not words:
                new_parts.append(part)
                continue
            new_words = []
            for word in words:
                if len(word) >= 3 and random.random() < 0.1:  # 30% chance for words 3+ letters
                    new_words.append(get_synonym_or_antonym(word))
                else:
                    new_words.append(word)  # Keep short words or 70% chance unchanged
            new_parts.append(" ".join(new_words))

    return "".join(new_parts)

def process_section(section):
    """Process a section, applying mangling to each line outside label/attribute."""
    lines = section.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith(("label:", "attribute:")):
            new_lines.append(line)
        else:
            new_lines.append(mangle_text(line))
    return "\n".join(new_lines)

def detect_encoding(file_path):
    """Detect file encoding by checking byte signature."""
    with open(file_path, "rb") as f:
        raw_bytes = f.read(4)
    if raw_bytes.startswith(b'\xFF\xFE'):
        return "utf-16-le"
    elif raw_bytes.startswith(b'\xFE\xFF'):
        return "utf-16-be"
    elif raw_bytes.startswith(b'\xEF\xBB\xBF'):
        return "utf-8-sig"
    else:
        try:
            with open(file_path, "r", encoding="utf-16-le") as f:
                f.read()
            return "utf-16-le"
        except:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read()
                return "utf-8"
            except:
                return "latin-1"

# Set input and output roots
source_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\modded_based"
output_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\synonym_antonym_mangled"

if not os.path.exists(source_root):
    print(f"Error: Source folder does not exist - {source_root}")
    sys.exit(1)

if not os.path.exists(output_root):
    os.makedirs(output_root)

# Collect all .txt files
txt_files = [os.path.join(root, file) for root, _, files in os.walk(source_root) for file in files if file.endswith(".txt")]
total_files = len(txt_files)

if total_files == 0:
    print(f"Error: No .txt files found in {source_root}")
    sys.exit(1)

print(f"Found {total_files} .txt files to process.")

# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file", dynamic_ncols=True) as pbar:
    for input_file_path in txt_files:
        relative_path = os.path.relpath(input_file_path, source_root)
        output_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        encoding = detect_encoding(input_file_path)
        print(f"Detected encoding for {os.path.basename(input_file_path)}: {encoding}")

        try:
            with open(input_file_path, "r", encoding=encoding) as f:
                raw_text = f.read()
        except Exception as e:
            print(f"Failed to read {os.path.basename(input_file_path)}: {e}")
            pbar.update(1)
            continue

        parts = raw_text.split("---", 1)
        header = parts[0].strip()
        if len(parts) <= 1:
            modded_text = header
        else:
            sections = parts[1].strip().split("\n---\n")
            mangled_sections = [process_section(section) for section in sections]
            modded_text = f"{header}\n---\n" + "\n---\n".join(mangled_sections)

        try:
            with open(output_file_path, "w", encoding=encoding) as f:
                f.write(modded_text)
            print(f"Wrote {os.path.basename(input_file_path)} to {output_file_path}")
        except Exception as e:
            print(f"Failed to write {os.path.basename(input_file_path)}: {e}")

        pbar.update(1)

print("Done! Files are now synonym/antonym mangled (3+ letters only) with logs in 'synonym_antonym_changes.log'.")