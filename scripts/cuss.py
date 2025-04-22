import os
import re
import random
import sys
import nltk
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

# Ensure NLTK resources are downloaded (run once: nltk.download('punkt'), nltk.download('averaged_perceptron_tagger'))
nltk.data.path.append(r"C:\nltk_data")  # Adjust if needed

# Your CUSS list from before, categorized loosely
CUSS = {
    "exclamations": ["frick", "dang", "heck", "darn", "shoot", "crud", "fudge", "jeez", "gosh", "drat",
                     "blast", "rats", "whoops", "oops", "yikes", "geez", "shucks", "cripes", "blimey", "snap",
                     "damn", "hell", "shit", "fuck", "piss", "holy", "crikey", "doggone", "dangit", "blastit",
                     "fucknuts", "shitting hell", "cockballs", "jesus tittyfucking christ", "arsebiscuits",
                     "bollockshit", "twatwaffles", "pissflaps", "goddamn motherfucker", "holy fucking shit",
                     "cuntbuckets", "arseblasting fuckery", "shitpiss", "cockwomble", "fannyfarts", "buggernuts",
                     "prickshits", "hellfuck", "dickdamn", "turdblast", "knobgoblins", "wankshaft", "sweet merciful crap",
                     "christ on a cracker", "holy mother of balls", "fucking hellfire", "shit on a shingle", "goddamn chaos",
                     "piss in a blender", "jesus wept", "fuck me sideways", "arse over tit", "bloody nora", "stone the crows",
                     "bugger me blind", "shit the bed", "holy roller", "fuck a duck", "piss and vinegar", "damn straight",
                     "hell’s bells", "cock and bull", "saints alive", "mother of pearl", "son of a gun", "blow me down",
                     "good grief", "lord have mercy", "what the shit", "holy smokes", "friggin’ aye", "balls to the wall",
                     "christ almighty", "hot damn", "well I’ll be damned", "fuckity fuck", "shitballs supreme",
                     "bugger all", "fuck my life", "holy crapola", "piss off already", "damn it to hell", "shit’s sake",
                     "jesus jumping jackrabbits", "fucksticks", "arseclowns unite", "holy flying fuck", "cocktastrophe",
                     "piss on a stick", "goddamn clusterfuck", "shit hits the fan", "fuck me running", "arse on fire",
                     "bloody hellscape", "crikey moses", "hell on wheels", "cock of the walk", "saints preserve us",
                     "mother of all fucks", "son of a bitchin’ hell", "blow my stack", "good lord almighty", "what in tarnation",
                     "holy guacamole", "friggin’ hellhole", "balls out bonanza", "christ on a bike", "hot shit damn",
                     "well slap my ass", "fuckaroo", "shitstorm deluxe", "arsepocalypse", "piss parade", "holy rollercoaster",
                     "damn near died", "hellfire and brimstone", "cockamamy nonsense", "fucktard fiesta", "shitshow spectacular",
                     "bloody bollocks", "fuck outta here", "holy schnikes", "piss in the wind", "damn the torpedoes",
                     "shit’s getting real", "jesus take the wheel", "fucknado", "arse end of nowhere", "holy hellstorm",
                     "cock and balls disaster", "piss up a rope", "goddamn trainwreck", "shitfaced and screaming",
                     "fuck me with a chainsaw", "arse about face", "bloody shambles", "crikey o’reilly", "hell in a handbasket",
                     "cock it up", "saints be praised", "motherfucking mayhem", "son of a whore", "blow the roof off",
                     "good god damn", "what the actual fuck", "holy mackerel", "friggin’ disasterpiece", "balls deep in bullshit",
                     "christ on crutches", "hot mess express", "well butter my butt", "fuckapalooza", "shit creek paddleless",
                     "arseclown convention", "piss poor planning", "holy handgrenade", "damn skippy", "hellbent for leather",
                     "cocked up beyond repair", "fuckwit free-for-all", "shitballs on fire", "arse kicking time",
                     "bloody ridiculous", "fuck me gently", "holy jumping jack", "piss in my cereal", "damn dirty apes",
                     "shit outta luck", "jesus h christ", "fucktacular mess", "arsehole central", "holy crap on a stick",
                     "cocked and loaded", "piss drunk madness", "goddamn goat rodeo", "shit sandwich supreme",
                     "fuck off forever", "arse over elbows", "bloody cockup central", "crikey almighty", "hell no way",
                     "cock sucking chaos", "saints forbid", "mother of madness", "son of a shitstorm", "blow it out your ass"],
    "nouns": ["ass", "bitch", "crap", "dick", "cock", "balls", "tits", "arse", "bollocks", "twat",
              "prick", "arsehole", "shite", "bugger", "fanny", "knob", "tosser", "wanker", "bellend",
              "git", "sod", "turd", "wang", "pussy", "jizz", "muff", "nads", "poon", "schlong", "skank",
              "spunk", "whore", "dickhead", "fuckwit", "shitbag", "cunt", "dipshit", "fucker", "jackass",
              "assclown", "shitlord", "cockgoblin", "twatnugget", "pisswizard", "arseweasel", "fuckface",
              "douchecanoe", "ballbag", "cumdumpster", "shitweasel", "prickface", "knobgobbler", "turdgoblin",
              "wankstain", "fannybandit", "cockjockey", "shitpickle", "buggermunch", "arsebadger", "spunkbubble",
              "dickwaffle", "cuntmuffin", "pissgargler", "fucktrumpet", "bollockbreath", "goon", "loon", "nutjob",
              "freak", "creep", "dolt", "tool", "chump", "goof", "muppet", "numpty", "plonker", "berk",
              "lout", "oaf", "clod", "dork", "nerd", "geek", "flake", "weirdo", "sucker", "loser", "dingus",
              "bonehead", "blockhead", "dimwit", "nitwit", "halfwit", "dumbass", "moron", "idiot", "lunatic",
              "madman", "psycho", "clusterfuck", "shitstorm", "trainwreck", "disaster", "fiasco", "cockup",
              "ballsack", "shitheap", "fuckery", "pisshole", "arsewipe", "dickcheese", "twatwaffle supreme",
              "arsehat", "cockmunch", "shitferbrains", "pissclown", "fuckbucket", "douchelord", "ballbreaker",
              "cumguzzler", "shitnado", "pricktease", "knobjockey", "turdblaster", "wankpuffin", "fannyflapper",
              "cockwaffle", "cuntbucket", "pissflipper", "fucknoodle", "bollockchops", "goofball", "nutcase",
              "whackjob", "oddity", "screwup", "dipstick", "buffoon", "clown", "joker", "twit", "dunce",
              "bozo", "lummox", "simpleton", "knucklehead", "pinhead", "airhead", "meathead", "shitwit",
              "fuckup", "mess", "wreck", "calamity", "debacle", "shitpile", "arsejam", "dickstorm", "pissfactory",
              "cockblocker", "twatlord", "fannywrecker", "buggerface", "wankmaster", "shitkicker", "douchewaffle",
              "arsebandit", "cockroach", "shitgibbon", "pissmonger", "fucktard", "douchebagel", "ballbuster",
              "cumrag", "shitshow", "prickwad", "knobend", "turdmuncher", "wanksock", "fannyfucker", "cockmaster",
              "cuntface", "pissbucket", "fuckstick", "bollockbrain", "goober", "nutter", "whacko", "freakazoid",
              "screwball", "dickwad", "buffoonery", "clownshow", "jokester", "twatface", "duncecap", "boob",
              "lunkhead", "pea-brain", "blockwit", "numbskull", "dope", "idiotface", "lunacy", "madness",
              "pandemonium", "mayhem", "havoc", "chaos", "shitheap", "fuckfest", "pisspond", "arsefactory",
              "cockknocker", "twatburger", "fannysmasher", "buggerwad", "wanklord", "shitstain", "douchepocalypse",
              "arsefucker", "cocktard", "shitfer", "pisslord", "fuckmonkey", "douchetrumpet", "ballfondler",
              "cumwaffle", "shitwreck", "prickmeister", "knobcheese", "turdwaffle", "wankbucket", "fannytrampler",
              "cockslapper", "cuntwaffle", "pissweasel", "fuckmuppet", "bollocknose", "goonbag", "nutbar",
              "whackdoodle", "oddball", "screwloose", "dipshit", "buffoonface", "clowncar", "jokebag", "twitface",
              "duncebucket", "bozohead", "lunk", "simplejack", "knucklebrain", "pinprick", "airbrain", "meatstick",
              "shitforbrains", "fuckmess", "messhall", "wreckage", "calamityjane", "debacletron", "shitshack",
              "arseplosion", "dickflood", "pissjungle", "cockblock", "twatking", "fannyfiend", "buggerboss",
              "wankwizard", "shitshoveler", "douchecanoeist", "arsepirate", "cockthief", "shitgoblin", "pissprince"],
    "adjectives": ["fucking", "shitty", "crappy", "pissy", "dickish", "cocky", "bloody", "arsey",
                   "wanky", "bollocksy", "twatty", "prickly", "shitey", "buggery", "fannied", "knobby",
                   "tosserish", "wankerish", "bellendish", "gittish", "soddy", "turdy", "pussyish",
                   "jizzy", "muffy", "nadish", "poony", "schlongy", "skanky", "spunky", "whorish",
                   "fucktastic", "shitfaced", "cocktacular", "pissawful", "arsemazing", "bollocktastic",
                   "twatacular", "prickish", "shiteous", "buggerfucked", "fannytastic", "knobular",
                   "tosstastic", "wankerrific", "bellendacious", "gittastic", "sodding", "turdalicious",
                   "pussyfied", "jizzalicious", "mufftastic", "nadiferous", "poonacious", "schlongerrific",
                   "skankalicious", "spunkerrific", "whoretastic", "fuckshitting", "cuntacular", "mental",
                   "bonkers", "nuts", "batshit", "loony", "crazy", "wild", "insane", "mad", "screwy",
                   "wacko", "zany", "daft", "nutsy", "barmy", "loopy", "goofy", "daffy", "silly",
                   "absurd", "ludicrous", "ridiculous", "hilarious", "mental", "freaky", "weird",
                   "odd", "bizarre", "stupid", "dumb", "thick", "dense", "brainless", "mindless",
                   "godawful", "hellish", "diabolical", "epic", "massive", "colossal", "titanic",
                   "shitastic", "pissified", "cockalicious", "arseworthy", "fucktabulous", "friggin",
                   "damnable", "hellacious", "cockamamie", "pissworthy", "arseholian", "bollockrific",
                   "twatastic", "prickalicious", "shiteful", "buggerlicious", "fannyriffic", "knobtastic",
                   "tosserrific", "wankalicious", "bellenderrific", "gitworthy", "soddalicious", "turdacious",
                   "pussywhipped", "jizzerrific", "muffalicious", "nadworthy", "poonerrific", "schlongalicious",
                   "skankerrific", "spunkalicious", "whoreworthy", "fucknificent", "cuntabulous", "nutty",
                   "whacked", "cuckoo", "berserk", "maniacal", "frantic", "rabid", "lunatic", "deranged",
                   "unhinged", "kooky", "quirky", "dippy", "flaky", "doddery", "scatterbrained", "asinine",
                   "preposterous", "outrageous", "farcical", "comical", "freakish", "peculiar", "twisted",
                   "moronic", "imbecilic", "cretinous", "dopey", "witless", "doltish", "catastrophic",
                   "monumental", "gargantuan", "shitacular", "pissendous", "cockerrific", "arsepalooza",
                   "fuckalicious", "damnworthy", "helltastic", "cockadoodle", "pissalicious", "arsefabulous",
                   "bollockworthy", "twatworthy", "prickerrific", "shitealicious", "buggeriffic", "fannyalicious",
                   "knoberrific", "tosseralicious", "wanktabulous", "bellendtabulous", "gitalicious", "sodtabulous",
                   "turderrific", "pussylicious", "jizzworthy", "mufferrific", "nadtabulous", "poonworthy",
                   "schlongtastic", "skanktabulous", "spunkworthy", "whoretabulous", "fuckerrific", "cuntastic",
                   "nutjobulous", "whackalicious", "cuckootastic", "berserkerrific", "maniacalicious", "frantabulous",
                   "rabiderrific", "lunatabulous", "derangerrific", "unhingetastic", "kookalicious", "quirktastic",
                   "dippalicious", "flakerrific", "dodderalicious", "scatterbraintastic", "asinitabulous",
                   "preposterrific", "outrageouserrific", "farcicalicious", "comicaltastic", "freakalicious",
                   "peculiartastic", "twistalicious", "morontastic", "imbecilerrific", "cretinouserrific",
                   "dopealicious", "witlesserrific", "doltalicious", "catastropherrific", "monumentalicious",
                   "gargantabulous", "shitstormerrific", "pissfloodalicious", "cockblocktastic", "arsejamerrific",
                   "fucknadoalicious", "damnhelltastic", "hellfuckerrific", "cockamamylicious", "pissfactorytastic",
                   "arsewreckalicious", "bollockshittastic", "twatfuckerrific", "prickshittabulous", "shitepissalicious",
                   "buggerfucktastic", "fannyfuckerrific", "knobpissalicious", "tosserfucktastic", "wankerfuckerrific",
                   "bellendshittastic", "gitpissalicious", "sodfucktastic", "turdpisserrific", "pussyfuckalicious",
                   "jizzshittastic", "mufffuckerrific", "nadshittabulous", "poonpissalicious", "schlongfucktastic",
                   "skankpisserrific", "spunkfuckalicious", "whoreshittastic", "fuckshitpisserrific", "cuntfucktastic"]
}

def get_cuss_replacement(word, pos):
    """Replace a word with a cuss word of the same part of speech, if possible."""
    if pos.startswith("JJ"):  # Adjectives
        return random.choice(CUSS["adjectives"]) if CUSS["adjectives"] else word
    elif pos.startswith("NN"):  # Nouns
        return random.choice(CUSS["nouns"]) if CUSS["nouns"] else word
    elif pos in ["UH", "VB", "RB"]:  # Exclamations, verbs, adverbs (rough match)
        return random.choice(CUSS["exclamations"]) if CUSS["exclamations"] else word
    return word  # Default to original if no match

def mangle_text(text):
    """Replace random words (3+ letters) with cuss words outside tags with 30% chance, preserving spacing."""
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
                if len(word) >= 3 and random.random() < 0.007:  # 30% chance for 3+ letter words
                    new_word = get_cuss_replacement(word, pos)
                    if new_word != word:  # Log only if changed
                        print(f"Changed '{word}' to '{new_word}' (POS: {pos})")
                        new_text = new_text[:start + offset] + new_word + new_text[end + offset:]
                        offset += len(new_word) - len(word)
            new_parts.append(new_text)

    return "".join(new_parts)

def process_section(section):
    """Process a section, applying cuss mangling to each line outside label/attribute."""
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
source_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\synonym_antonym_mangled"
output_root = r"C:\Users\jesse\Desktop\em\NEW_WORDS\cuss_mangled"

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

print("Done! Files are now cuss-word mangled in the output folder.")