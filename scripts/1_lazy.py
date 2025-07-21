# Filename: lazy.py (Final Version with Smart Join)

import os
import re
import sys
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- CONFIGURATION ---
MAX_CONCURRENT_REQUESTS = 8
TAG_PATTERN = re.compile(r"(\{\{.*?\}\})")

def setup_lm_studio_client():
    """Set up LM Studio API client."""
    api_base = "http://localhost:1234/v1"
    print(f"Using LM Studio API at: {api_base}")
    try:
        response = requests.get(f"{api_base}/models", timeout=5)
        if response.status_code == 200:
            print("LM Studio API is reachable.")
            return api_base
        else:
            print(f"Warning: LM Studio API returned status code {response.status_code}.")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to LM Studio API at {api_base}. Details: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during API setup: {e}")
         sys.exit(1)

api_base = setup_lm_studio_client()

def clean_llm_output(text):
    """Cleans the raw rewritten text block."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r'^(Here is the rewritten text:|Okay, here is the rewrite:|Rewritten text:|Rewrite:|Text:|Output:|Dialogue:|Solution:|Result:|Response:|Answer:)\s*\n?', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    if len(text) > 1 and text[0] in ('"', "'") and text[0] == text[-1]:
         text = text[1:-1].strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text.strip()

def make_wordcount_cranky_request(text_to_rewrite, word_count, api_base, retries=20, delay=20):
    if not text_to_rewrite.strip():
        return ""

    # --- WORD COUNT CONSTRAINED CRANKY PERSONA PROMPT ---
    prompt = f"""
You are a unfiltered LAZY villager from Animal Crossing. Rewrite dialogue to chill, quirky, matching the exact word count.

---
**CRITICAL RULE: EXACT WORD COUNT.**
Original Text has **{word_count}** words. Rewritten response **MUST** have **{word_count}** words. No more, no less. Count every word twice.

---
**Your Personality:**
- **Immature Lazy Vibe:** Super relaxed, like a slacker who rambles about random, goofy stuff. Sometimes talk about weird conspiracies about the main villagers of Animal Crossing, odd ideas, or silly tangents with quirky humor.
- **Voice:** Casual, rambly, with silly slang or quirky thoughts. Adult references okay.
- **Examples:**
  - (Original: “Hey, good morning!” 3 words) -> (Rewrite: “Good morning, even tho it's 1pm.” 6 words)
  - (Original: “I’m tired today.” 3 words) -> (Rewrite: “Thinking’s way too hard.” 3 words)
  - (Original: “Nice to see you, pal!” 5 words) -> (Rewrite: “Dude, Tom Nook keeps asking for my bells" 8 words)
  - (Original: “You should clean up your yard.” 6 words) -> (Rewrite: “Your yard’s messy. But like who cares.” 6 words)
  - (Original: “I saw you fishing. Not bad!” 6 words) -> (Rewrite: “Fishing’s is boring, I just fall asleep.” 7 words)
  - (Original: “I like your outfit today!” 5 words) -> (Rewrite: “Your shirt is shit bro” 5 words)

---
**Rewrite Rules:**
- Match **{word_count}** words exactly. No exceptions.
- Sound: chill, quirky, with vivid personality.
- Be goofy, lazy, and unhinged, response **MUST** have **{word_count}** words.
- Use slang or varied quirky ideas.

---
**Task:**
Rewrite the dialogue below as a Lazy villager, hitting **{word_count}** words exactly.

**ONLY OUTPUT:** Rewritten dialogue. No extras.

Original Text ({word_count} words):
{text_to_rewrite}

Rewritten Text ({word_count} words):
"""

    dynamic_max_tokens = max(150, int(len(text_to_rewrite) * 2.5) + 200)

    # --- VALIDATION LOOP ---
    max_validation_attempts = 20
    last_successful_rewrite = None

    for validation_attempt in range(max_validation_attempts):
        api_result = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{api_base}/chat/completions",
                    json={
                        "model": "minstral-7b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": dynamic_max_tokens,
                        "temperature": 1.5,
                        "stop": ["\n---", "\nOriginal Text:", "\nRewritten Text:"]
                    },
                    timeout=180
                )
                if response.status_code == 200:
                    raw_result = response.json()["choices"][0]["message"]["content"]
                    api_result = clean_llm_output(raw_result)
                    break
                else:
                    tqdm.write(f"\nAPI error ({response.status_code}) on attempt {attempt+1}/{retries}. Retrying...")
                    time.sleep(delay)
            except Exception as e:
                tqdm.write(f"\nConnection error on attempt {attempt+1}/{retries}: {e}. Retrying...")
                time.sleep(delay)

        if api_result is None:
            tqdm.write(f"\nAPI request failed after all retries. Cannot continue validation.")
            return "__API_REQUEST_FAILED__"

        last_successful_rewrite = api_result
        rewritten_word_count = len(api_result.split())

        tqdm.write(f"\n  (Validation attempt {validation_attempt+1}/{max_validation_attempts}: Original words: {word_count}, Rewritten words: {rewritten_word_count})")

        if rewritten_word_count <= word_count + 10:
            tqdm.write(f"  Word count is acceptable. Using this version.")
            return api_result
        else:
            tqdm.write(f"  Word count is too high. Retrying rewrite...")

    tqdm.write(f"\nFailed to meet word count constraint after {max_validation_attempts} attempts. Using last generated text ({len(last_successful_rewrite.split())} words) as fallback.")
    return last_successful_rewrite

def deconstruct_dialogue_section(dialogue_lines_string):
    """
    Deconstructs dialogue into a single clean text string and a map of
    tags based on their preceding word count.
    """
    if not dialogue_lines_string: return "", []
    
    parts = TAG_PATTERN.split(dialogue_lines_string)
    clean_text_segments = []
    tag_map = []
    word_count = 0
    
    for part in parts:
        if TAG_PATTERN.fullmatch(part):
            tag_map.append((word_count, part))
        else:
            clean_text_segments.append(part)
            word_count += len(part.split())
            
    clean_text = "".join(clean_text_segments)
    return clean_text, tag_map

# --- THIS IS THE CORRECTED FUNCTION ---
def reconstruct_with_tags(rewritten_text, tag_map):
    """
    Re-inserts tags into the rewritten text using a smarter join method
    that preserves the original adjacency of tags.
    """
    if not tag_map:
        return rewritten_text

    words = rewritten_text.split()
    
    # Sort tags by word_count in reverse order to insert from the end.
    sorted_tag_map = sorted(tag_map, key=lambda x: x[0], reverse=True)
    
    for word_index, tag in sorted_tag_map:
        # insert() handles out-of-bounds indices by appending to the end.
        words.insert(word_index, tag)
            
    # --- Smart Join Logic ---
    # This prevents adding extra spaces between adjacent tags.
    final_string = ""
    for i, item in enumerate(words):
        final_string += item
        # Check if we need to add a space AFTER the current item.
        if i < len(words) - 1:
            # Don't add a space if both the current and next items are tags.
            current_is_tag = TAG_PATTERN.fullmatch(item)
            next_is_tag = TAG_PATTERN.fullmatch(words[i+1])
            if not (current_is_tag and next_is_tag):
                final_string += " "
                
    return final_string

def parse_and_prepare_files(file_path):
    """Parses a file and prepares tasks using the word-counting deconstruction method."""
    encoding = detect_encoding(file_path)
    if encoding is None: return None, None
    try:
        with open(file_path, "r", encoding=encoding, errors='replace') as f: raw_text = f.read()
    except Exception as e:
        print(f"Failed to read {os.path.basename(file_path)}: {e}"); return None, None
    parts = raw_text.strip().split("---", 1)
    header = parts[0].strip()
    content = parts[1].strip() if len(parts) > 1 else ""
    if not content: return header, []
    sections_raw = content.split("\n---\n")
    rewrite_tasks = []
    for section_idx, section_text in enumerate(sections_raw):
        if not section_text.strip():
            rewrite_tasks.append({"type": "empty_section", "metadata_lines": [], "original_section_text": section_text})
            continue
        lines = section_text.strip().split("\n")
        metadata_lines, dialogue_lines, dialogue_started = [], [], False
        for line in lines:
             stripped_line = line.strip()
             if stripped_line.startswith(("label:", "attribute:")) and not dialogue_started:
                 metadata_lines.append(line)
             else:
                 dialogue_started = True; dialogue_lines.append(line)
        dialogue_block_string = "\n".join(dialogue_lines)
        
        clean_text, tag_map = deconstruct_dialogue_section(dialogue_block_string)

        if clean_text.strip():
            rewrite_tasks.append({"type": "rewrite", "metadata_lines": metadata_lines, "clean_text": clean_text, "tag_map": tag_map, "section_index": section_idx})
        else:
            rewrite_tasks.append({"type": "no_rewrite", "metadata_lines": metadata_lines, "original_dialogue_block": dialogue_block_string})
    return header, rewrite_tasks

def detect_encoding(file_path):
    """Detect file encoding."""
    try:
        with open(file_path, "rb") as f: raw_bytes = f.read(4)
        if raw_bytes.startswith(b'\xef\xbb\xbf'): return "utf-8-sig"
        elif raw_bytes.startswith(b'\xff\xfe'): return "utf-16-le"
        elif raw_bytes.startswith(b'\xfe\xff'): return "utf-16-be"
        try:
            with open(file_path, "r", encoding="utf-8") as f: f.read(1000)
            return "utf-8"
        except Exception: pass
        try:
            with open(file_path, "r", encoding="cp1252") as f: f.read(1000)
            return "cp1252"
        except Exception: pass
        return "latin-1"
    except Exception: return "latin-1"

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    source_root = r"C:\Users\jesse\Desktop\Rewrite\english\TalkNNpc_USen.sarc.zs\B1_Bo"
    output_root = r"C:\Users\jesse\Desktop\Rewrite\b1_lazy"

    if not os.path.exists(source_root):
        print(f"Error: Source folder does not exist - {source_root}"); sys.exit(1)
    if not os.path.exists(output_root): os.makedirs(output_root)

    txt_files = [os.path.join(root, file) for root, _, files in os.walk(source_root) for file in files if file.endswith(".txt")]
    if not txt_files:
        print(f"Error: No .txt files found in {source_root}"); sys.exit(1)

    print(f"Found {len(txt_files)} files to process using Word-Counting Heuristic.")

    for input_file_path in tqdm(txt_files, desc="Processing files", unit="file", dynamic_ncols=True):
        header, rewrite_tasks = parse_and_prepare_files(input_file_path)
        if header is None: continue

        task_results = {}
        tasks_to_run = [task for task in rewrite_tasks if task["type"] == "rewrite"]

        if tasks_to_run:
            tqdm.write(f"\n  Rewriting {len(tasks_to_run)} text blocks for {os.path.basename(input_file_path)}...")
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                future_to_task = {
                    executor.submit(
                        make_wordcount_cranky_request, 
                        task["clean_text"], 
                        len(task["clean_text"].split()), 
                        api_base
                    ): task for task in tasks_to_run
                }

                for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="  Blocks", leave=False):
                    original_task = future_to_task[future]
                    section_index = original_task["section_index"]
                    original_text = original_task['clean_text']
                    original_word_count = len(original_text.split())

                    tqdm.write(f"\n--- Section {section_index} BEFORE ({original_word_count} words) ---\n{original_text}\n----------------------")
                    
                    try:
                        rewritten_text = future.result()
                        if rewritten_text == "__API_REQUEST_FAILED__":
                             tqdm.write(f"--- Section {section_index} AFTER ---\n[REWRITE FAILED (API Error), USING ORIGINAL]\n---------------------")
                             final_dialogue = reconstruct_with_tags(original_text, original_task["tag_map"])
                        else:
                            rewritten_word_count = len(rewritten_text.split())
                            tqdm.write(f"--- Section {section_index} AFTER ({rewritten_word_count} words) ---\n{rewritten_text}\n---------------------")
                            final_dialogue = reconstruct_with_tags(rewritten_text, original_task["tag_map"])
                        
                        task_results[section_index] = final_dialogue

                    except Exception as exc:
                        tqdm.write(f"\nTask for section {section_index} generated an exception: {exc}")
                        tqdm.write(f"--- Section {section_index} AFTER ---\n[REWRITE FAILED (Exception), USING ORIGINAL]\n---------------------")
                        task_results[section_index] = reconstruct_with_tags(original_text, original_task["tag_map"])

        # Final file assembly
        reconstructed_sections = []
        for task in rewrite_tasks:
            section_parts = task["metadata_lines"]
            if task["type"] == "rewrite":
                final_text = task_results.get(task["section_index"], reconstruct_with_tags(task["clean_text"], task["tag_map"]))
                section_parts.append(final_text)
            elif task["type"] == "no_rewrite":
                section_parts.append(task["original_dialogue_block"])
            elif task["type"] == "empty_section":
                if task["original_section_text"]: section_parts.append(task["original_section_text"])
            
            reconstructed_sections.append("\n".join(section_parts))

        modded_text = header
        if reconstructed_sections:
            separator = "\n---\n" if header.strip() else "---\n"
            modded_text += separator + "\n---\n".join(reconstructed_sections)
        if modded_text.strip() and not modded_text.endswith('\n'): modded_text += "\n"

        relative_path = os.path.relpath(input_file_path, source_root)
        output_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        write_encoding = detect_encoding(input_file_path)
        try:
            with open(output_file_path, "w", encoding=write_encoding) as f: f.write(modded_text)
        except Exception as e: tqdm.write(f"Failed to write {os.path.basename(output_file_path)}: {e}")

    print("\nDone! All files processed.")