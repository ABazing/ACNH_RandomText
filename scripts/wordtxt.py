import os
import re
import random
import sys
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

CUSS = [
    # Mild Exclamations (~100+)
    "frick", "dang", "heck", "darn", "shoot", "crud", "fudge", "jeez", "gosh", "drat",
    "blast", "rats", "whoops", "oops", "yikes", "geez", "shucks", "cripes", "blimey", "snap",
    "zounds", "egads", "golly", "sheesh", "phew", "whew", "cor", "blazes", "jiminy", "phooey",
    "bother", "dash", "fiddlesticks", "goodness", "gracious", "heavens", "myword", "nuts", "poot", "sakes",
    "strewth", "tarnation", "toot", "whiz", "yowza", "zoinks", "bingo", "eureka", "huzzah", "oy",
    "gadzooks", "holy", "crikey", "doggone", "dangit", "blastit", "whammo", "kapow", "zowie", "pow",
    "bosh", "fie", "pshaw", "bah", "humbug", "horsefeathers", "poppycock", "balderdash", "hogwash", "malarkey",
    "nertz", "phoo", "pish", "posh", "rot", "rubbish", "tosh", "twaddle", "bunk", "fiddle",
    "flap", "flimflam", "fooey", "gah", "gee", "ha", "ho", "hokey", "jeepers", "lawd",
    "lordy", "migosh", "ohmy", "ouch", "ow", "shazam", "shoo", "splat", "ugh", "whee",
    "whoa", "wowsers", "yeesh", "yippee", "zap", "zing", "zounds", "zut", "oi", "argh",

    # Traditional Cuss Words (~150+)
    "damn", "hell", "shit", "fuck", "ass", "bitch", "crap", "piss", "dick", "cock",
    "balls", "tits", "arse", "wank", "bollocks", "twat", "prick", "arsehole", "shite", "bugger",
    "fanny", "knob", "tosser", "wanker", "bellend", "git", "sod", "turd", "wang", "pussy",
    "jizz", "muff", "nads", "poon", "schlong", "skank", "spunk", "whore", "dickhead", "fuckwit",
    "shitbag", "cunt", "dipshit", "fucker", "jackass", "pissflap", "shitlord", "cockwaffle", "dumbass",
    "fuckstick", "shitgibbon", "twatwaffle", "douche", "fuckface", "asshat", "bullshit", "pricktease", "shitheel",
    "cockmunch", "arsewipe", "bastard", "bloody", "chode", "clit", "cum", "douchebag", "fart", "felch",
    "gash", "minge", "nonce", "pecker", "pissbag", "prat", "puke", "rimjob", "scrote", "slag",
    "slut", "smeg", "snatch", "suck", "taint", "toolbag", "tramp", "wazzock", "boner", "buttfuck",
    "clusterfuck", "cockbag", "crapsack", "fucknugget", "pissweasel", "shitface", "thundercunt", "wankstain",
    "assclown", "bint", "dickwad", "fucktard", "shart", "asswipe", "cocknob", "cumdump", "fuckhole",
    "pisswit", "shitstain", "titfuck", "arsebandit", "ballbag", "cockend", "cumbubble", "dickcheese",
    "fartknocker", "fuckbucket", "pisshead", "shitbreath", "twatface", "arsebreath", "cockjockey", "crapweasel",
    "dickbreath", "fuckknob", "pissgoblin", "shitmunch", "titwank", "arseclown", "cockpiss", "cumfart",
    "dickshit", "fuckarse", "pissclown", "shitdick", "twatshit", "arsefuck", "cockshite", "cuntface",
    "dickfart", "fuckpiss", "pissfuck", "shitarse", "twatcock", "arsepiss", "cocktwat", "cuntshit",

    # Insults (~200+)
    "jerk", "loser", "idiot", "moron", "fool", "clown", "dork", "nerd", "goof", "twit",
    "numbskull", "bonehead", "dimwit", "dope", "lamebrain", "nitwit", "blockhead", "dweeb", "creep", "sleaze",
    "weirdo", "goon", "buffoon", "slob", "pig", "rat", "snake", "tool", "airhead", "brat",
    "chump", "dingbat", "dolt", "dunce", "flake", "gimp", "halfwit", "klutz", "loon", "muppet",
    "oaf", "peon", "pleb", "schmuck", "scum", "simpleton", "sissy", "stooge", "sucker", "twerp",
    "wimp", "yokel", "zero", "boob", "bozo", "crank", "dickbag", "drongo", "grub", "knucklehead",
    "lout", "meathead", "pinhead", "plonker", "pud", "punk", "rube", "sap", "schlub", "scrub",
    "shmo", "slacker", "spaz", "wuss", "bungler", "churl", "codger", "cretin", "drip", "dud",
    "duffer", "dullard", "galoot", "gawk", "goober", "hayseed", "hick", "jerkoff", "laggard", "lummox",
    "mook", "nimrod", "noodle", "numpty", "pansy", "pillock", "pissant", "poltroon", "putz", "schmo",
    "schnook", "slime", "snot", "stinker", "stupe", "thicko", "wally", "weakling", "whiner", "zilch",
    "addlepate", "barbarian", "blunderer", "boor", "bumpkin", "cad", "coward", "crab", "cur", "dawdler",
    "dickweed", "dill", "dingle", "dodo", "drool", "dudley", "dung", "fink", "flunky", "gob",
    "gomer", "goofball", "goose", "grouch", "grump", "hob", "hobo", "ignoramus", "jackanapes", "jelly",
    "jinx", "jughead", "kooky", "leech", "lily", "lunk", "maggot", "malingerer", "maw", "mope",
    "ninny", "nutter", "oddity", "pest", "piffle", "pimple", "poophead", "prude", "riffraff", "rotter",
    "runt", "sack", "scab", "scallywag", "screwball", "shirk", "skunk", "slouch", "snooze", "soggy",
    "sot", "sourpuss", "spud", "squit", "stale", "stinkpot", "stodge", "swab", "tadpole", "toad",
    "trifle", "turkey", "vandal", "vermin", "waddle", "waffler", "whelp", "worm", "yahoo", "zit",

    # Compliments (~150+)
    "sweetie", "champ", "buddy", "pal", "mate", "darling", "honey", "cutie", "babe", "dear",
    "star", "hero", "genius", "ace", "pro", "legend", "rockstar", "winner", "beauty", "gem",
    "treasure", "angel", "sunshine", "sport", "kiddo", "chief", "boss", "dude", "friend", "amigo",
    "bestie", "bravo", "captain", "crackerjack", "doll", "dreamboat", "firecracker", "gold", "great", "guru",
    "heart", "hotshot", "icon", "jewel", "king", "lady", "lord", "love", "maestro", "marvel",
    "master", "maverick", "peach", "pearl", "prince", "princess", "queen", "rose", "sage", "saint",
    "scholar", "spark", "stud", "sugar", "superstar", "tiger", "topdog", "trooper", "unicorn", "vip",
    "visionary", "warrior", "whiz", "wonder", "admiral", "allstar", "beacon", "bigshot", "blessing", "bloom",
    "brain", "brilliant", "catch", "charmer", "cheer", "comet", "coolcat", "crown", "dandy", "delight",
    "diamond", "diva", "dynamite", "eagle", "elite", "emerald", "fame", "fancy", "flash", "flower",
    "force", "glory", "golden", "grace", "grand", "honcho", "hope", "idol", "inspiration", "jazz",
    "joy", "knight", "luminary", "magic", "magnate", "miracle", "muse", "nifty", "noble", "oracle",
    "paragon", "phoenix", "pinnacle", "pioneer", "platinum", "prize", "radiance", "rarity", "rebel", "ruby",
    "savior", "shine", "silver", "siren", "skipper", "smasher", "snapper", "spunk", "stunner", "talent",
    "titan", "trailblazer", "trump", "valor", "velvet", "vibe", "victor", "vixen", "wizard", "zest"
]
def setup_m2m100_translator():
    """Load M2M-100 model and tokenizer with FP16 for efficiency."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "facebook/m2m100_418M"
    print(f"Loading model: {model_name}")

    try:
        model = M2M100ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        sys.exit(1)

    return model, tokenizer, device

# Initialize M2M-100
model, tokenizer, device = setup_m2m100_translator()

def add_cussing(text):
    """Add cuss words at 5% chance per word, only to non-tag text."""
    if not text.strip():
        return text
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
                new_words.append(word)
                if random.random() < 0.05:
                    cuss = random.choice(CUSS)
                    cuss = cuss.lower() if random.random() > 0.01 else cuss.upper()
                    new_words.append(cuss)
            new_parts.append(" ".join(new_words))
    return "".join(new_parts)

def m2m100_translate_mangle(texts):
    """Batch translate texts through 4 hops (EN -> Random1 -> Random2 -> Random3 -> EN), only non-tag parts."""
    if not texts or all(not t.strip() for t in texts):
        return texts

    # Apply cussing to all texts first
    cussed_texts = [add_cussing(text) for text in texts]

    # Split texts into parts and track translatable segments per line
    tag_pattern = r"(\{\{.*?\}\})"
    all_parts = [re.split(tag_pattern, text) for text in cussed_texts]
    translatable_texts = []
    part_mappings = []  # Store (line_idx, part_idx) for each translatable part

    for line_idx, parts in enumerate(all_parts):
        for part_idx, part in enumerate(parts):
            if not (part.startswith("{{") and part.endswith("}}")) and part.strip():
                # Stabilize single characters
                translatable_part = part.strip()
                if len(translatable_part) == 1 and translatable_part in "!.?,":
                    translatable_part += " "
                translatable_texts.append(translatable_part)
                part_mappings.append((line_idx, part_idx))

    # If nothing to translate, return cussed texts
    if not translatable_texts:
        return cussed_texts

    max_length = 128
    truncated_texts = [" ".join(t.split()[:max_length]) if len(t.split()) > max_length else t for t in translatable_texts]

    try:
        current_texts = truncated_texts
        languages = [
            "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar",
            "hi", "bn", "vi", "th", "tr", "pl", "cs", "nl", "sv", "el"
        ]
        hop_chain = random.sample(languages, 3)
        print(f"Batch translating: EN -> {hop_chain[0]} -> {hop_chain[1]} -> {hop_chain[2]} -> EN")
        print(f"Translatable inputs: {truncated_texts}")

        with torch.no_grad():
            # Hop 1: EN -> Random1
            tokenizer.src_lang = "en"
            inputs = tokenizer(current_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(hop_chain[0]),
                                           max_length=128, num_beams=3, no_repeat_ngram_size=3, early_stopping=True)
            current_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            print(f"EN -> {hop_chain[0]}: {current_texts}")

            # Hop 2: Random1 -> Random2
            tokenizer.src_lang = hop_chain[0]
            inputs = tokenizer(current_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(hop_chain[1]),
                                           max_length=128, num_beams=3, no_repeat_ngram_size=3, early_stopping=True)
            current_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            print(f"{hop_chain[0]} -> {hop_chain[1]}: {current_texts}")

            # Hop 3: Random2 -> Random3
            tokenizer.src_lang = hop_chain[1]
            inputs = tokenizer(current_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(hop_chain[2]),
                                           max_length=128, num_beams=3, no_repeat_ngram_size=3, early_stopping=True)
            current_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            print(f"{hop_chain[1]} -> {hop_chain[2]}: {current_texts}")

            # Hop 4: Random3 -> EN
            tokenizer.src_lang = hop_chain[2]
            inputs = tokenizer(current_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("en"),
                                           max_length=128, num_beams=3, no_repeat_ngram_size=3, early_stopping=True)
            mangled_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            print(f"{hop_chain[2]} -> EN: {mangled_texts}")

        # Reassemble with original tags
        final_texts = cussed_texts.copy()
        for (line_idx, part_idx), mangled_text in zip(part_mappings, mangled_texts):
            parts = all_parts[line_idx]
            # Remove stabilizer space if added
            if mangled_text.endswith(" ") and len(mangled_text) > 1 and mangled_text[-2] in "!.?,":
                mangled_text = mangled_text[:-1]
            parts[part_idx] = mangled_text
            final_texts[line_idx] = "".join(parts)

        torch.cuda.empty_cache()
        return final_texts

    except Exception as e:
        print(f"M2M-100 batch translation error: {e} - returning original texts")
        return cussed_texts

def process_section(section, batch_size=8):
    """Process section with batched translation."""
    lines = section.split("\n")
    new_lines = lines.copy()  # Start with all lines
    translatable_lines = []
    line_indices = []

    # Collect lines to translate
    for i, line in enumerate(lines):
        if not line.strip().startswith(("label:", "attribute:")):
            translatable_lines.append(line)
            line_indices.append(i)

    # Batch translate
    if translatable_lines:
        mangled_lines = []
        for i in range(0, len(translatable_lines), batch_size):
            batch = translatable_lines[i:i + batch_size]
            mangled_batch = m2m100_translate_mangle(batch)
            mangled_lines.extend(mangled_batch)

        # Replace translatable lines with mangled versions
        for idx, mangled_line in zip(line_indices, mangled_lines):
            new_lines[idx] = mangled_line

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

# Set fixed input and output roots
source_root = r"C:\Users\jesse\Desktop\em\dialogue mod\Default txt\10"
output_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\modded"

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

print("Done! Convert .txt files from the output folder back to .msbt.")