import os
import re
import random
import sys
import nltk
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

# Ensure NLTK resources are downloaded (run once: nltk.download('punkt'), nltk.download('averaged_perceptron_tagger'))
nltk.data.path.append(r"C:\nltk_data")  # Adjust if needed

# Your wordz list from before, categorized loosely

wordz = {
    "exclamations": [],
    "nouns": [],
    "adjectives": []
}

def get_wordz_replacement(word, pos):
    """Replace a word with a wordz word of the same part of speech, if possible."""
    if pos.startswith("JJ"):  # Adjectives
        return random.choice(wordz["adjectives"]) if wordz["adjectives"] else word
    elif pos.startswith("NN"):  # Nouns
        return random.choice(wordz["nouns"]) if wordz["nouns"] else word
    elif pos in ["UH", "VB", "RB"]:  # Exclamations, verbs, adverbs (rough match)
        return random.choice(wordz["exclamations"]) if wordz["exclamations"] else word
    return word  # Default to original if no match

def mangle_text(text):
    """Replace random words (3+ letters) with wordz words outside tags with 30% chance, preserving spacing."""
    if not text.strip():
        return text

    # Split around tags
    parts = re.split(r"(\{\{.*?\}\})", text)
    new_parts = []

    for part in parts:
        if part.startswith("{{") and part.endswith("}}"):
            new_parts.append(part)  # Keep tags unchanged
        else:
            # Tokenize and tag words
            words = word_tokenize(part)
            tagged_words = pos_tag(words)
            # Use regex to find word boundaries in the original text
            word_pattern = re.compile(r'\b\w+\b')
            word_matches = [(m.group(), m.start(), m.end()) for m in word_pattern.finditer(part)]

            # Map tokenized words to regex matches (approximate)
            new_text = part
            offset = 0
            for (word, start, end), (tagged_word, pos) in zip(word_matches, tagged_words):
                if len(word) >= 3 and random.random() < 0.05:  # 30% chance for 3+ letter words
                    new_word = get_wordz_replacement(word, pos)
                    if new_word != word:  # Log only if changed
                        print(f"Changed '{word}' to '{new_word}' (POS: {pos})")
                        new_text = new_text[:start + offset] + new_word + new_text[end + offset:]
                        offset += len(new_word) - len(word)
            new_parts.append(new_text)

    return "".join(new_parts)

def process_section(section):
    """Process a section, applying wordz mangling to each line outside label/attribute."""
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
source_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\modded_en"
output_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\wordz_mangled"

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

print("Done! Files are now wordz-word mangled in the output folder.")