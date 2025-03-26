import os
import random
import re
import sys
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Nonsense banks (same size, used sparingly)
NONSENSE_BASIC = [
    "hey", "yo", "what", "cool", "dude", "man", "yeah", "nope", "okay", "sure", "wow", "nice", "ugh", "fine", "hello", "bye", "see", "look", "gotcha", "nah", "yep", "maybe", "dang", "oops", "whoa", "chill", "sweet", "lame", "deal", "go", "stop", "wait", "huh", "oh", "yay", "boo", "meh", "like", "love", "hate", "good", "bad", "big", "small", "fast", "slow", "hot", "cold", "up", "down", "left", "right", "here", "there", "now", "later", "soon", "done", "start", "end", "eat", "drink", "sleep", "walk", "run", "sit", "stand", "talk", "shut", "open", "close", "grab", "drop", "push", "pull", "hit", "kick", "throw", "catch", "jump", "fall", "stay", "leave", "come", "back", "out", "in", "on", "off", "over", "under", "around", "through", "yes", "no", "please", "thanks", "sorry", "ouch", "ew", "gross", "fun", "weird", "crazy", "neat", "dumb", "smart", "easy", "hard", "loud", "quiet", "soft", "tough", "weak", "high", "low", "old", "new", "wet", "dry", "clean", "dirty", "full", "empty", "more", "less", "all", "some", "none", "mine", "yours", "ours", "this", "that", "these", "those", "where", "when", "why", "how", "who", "which", "gimme", "take", "give", "need", "want", "got", "lost", "found", "check", "bet", "guess", "hope", "yup", "kinda", "sorta", "dunno", "whatever", "anyway", "alright", "awesome", "rad", "sick", "lit", "vibe", "mood", "bro", "fam", "pal", "buddy", "kid", "guy", "girl", "folk", "peeps", "home", "work", "play", "rest", "move", "hang", "chat", "call", "text", "ask", "tell", "say", "hear", "listen", "watch", "feel", "touch", "smell", "taste", "think", "know", "forget", "remember", "try", "fail", "win", "lose", "pick", "choose", "cut", "break", "fix", "make", "build", "wreck", "mess", "clean", "cook", "bake", "burn", "freeze", "melt", "shake", "stir", "flip", "roll", "slide", "spin", "turn", "twist", "bend", "stretch", "squeeze", "pop", "crack", "snap", "bang", "buzz", "ring", "beep", "honk", "clap", "cheer", "laugh", "cry", "shout", "whisper", "sing", "dance", "wave", "point", "nod", "hug", "kiss", "wink", "smile", "frown", "glare", "stare", "yawn", "sneeze", "cough", "hiccup", "fart", "pee", "poop", "sweat", "itch", "scratch", "hurt", "heal", "well", "tired", "awake", "dream", "wake", "nap", "snore", "drool", "trip", "slip", "crash", "bump", "smash", "bash", "whack", "slap", "punch", "bite", "chew", "swallow", "spit", "lick", "suck", "blow", "sip", "gulp", "chug", "spill", "drip", "wipe", "rub", "pat", "tap", "poke", "prod", "tickle", "pinch", "hold", "toss", "miss", "score", "beat", "race", "chase", "hide", "seek", "find", "tie", "dare", "joke", "tease", "prank", "trick", "fool", "lie", "truth", "swear", "promise", "wish", "plan", "rush", "relax", "stress", "freak", "calm", "wild", "tame", "bright", "dark", "shiny", "dull", "smooth", "rough", "sharp", "blunt", "heavy", "light", "huge", "tiny", "long", "short", "wide", "narrow", "deep", "shallow", "near", "far", "early", "late",
    "death", "killer", "murder", "doom", "grave", "stab", "blood", "gore", "rot", "skull", "bone", "ghost", "haunt", "creep", "slay", "gloom", "drown", "choke", "bleed", "slash", "tomb", "crypt", "fear", "scream", "die", "kill", "pain", "void", "curse", "wound", "grim", "shadow", "spook", "thrash", "plague", "venom", "claw", "tear", "crush", "ash", "mourn", "rage", "horror", "panic", "dread", "evil", "hell", "zombie", "fang", "howl", "shriek", "gasp", "sob", "trap", "chain", "bury", "guts", "flesh", "reek", "mold", "rust", "crumble", "agony", "torment", "scar", "bruise", "lash", "shatter", "explode", "flame", "smoke", "strangle", "noose", "plunge", "sink", "fade", "black", "ice", "shiver", "quake", "blind", "lurk", "stalk", "hunt", "prey", "snare", "seize", "yank", "drag", "twitch", "spasm", "writhe", "flail", "pound", "batter", "ruin", "fate", "hex", "bane", "woe", "grief", "sting", "ache", "throb", "gush", "spurt", "drench", "swamp", "mire", "retch", "puke", "slobber", "froth", "burst", "thunder", "roar", "wail", "moan", "whine", "whimper", "tear", "plummet", "scorch", "blaze", "flicker", "dim", "cease", "stiff", "silent", "mute", "dead", "dust", "vapor", "mist", "missing", "hidden"
]
NONSENSE_WTF = [
    "boop", "bloop", "blarp", "zwoosh", "kapow", "wham", "bzzzt", "plink", "plonk", "thwack", "splat", "splosh", "whoop", "whizz", "bang", "clang", "clunk", "ding", "dong", "fwoop", "glorp", "gloop", "honk", "hoot", "kerplunk", "moo", "meow", "rawr", "roar", "scree", "shloop", "sizzle", "snap", "snort", "squawk", "squelch", "thump", "toot", "twang", "vroom", "wheeze", "whistle", "woof", "zap", "zing", "zorch", "bawk", "beep", "blip", "bonk", "burp", "chomp", "clack", "click", "clop", "crunch", "drip", "fizz", "flap", "flop", "grunt", "gurgle", "hiss", "howl", "jingle", "klank", "knock", "meep", "mrrp", "nudge", "oink", "ping", "plop", "poof", "pop", "purr", "quack", "rasp", "rattle", "ribbit", "rumble", "scoot", "scratch", "shriek", "skree", "slap", "slosh", "slurp", "smack", "sniff", "snoot", "splash", "squeak", "squeal", "squish", "swish", "thud", "tick", "tock", "trill", "tweet", "vwoom", "wack", "whack", "whirr", "whoosh", "womp", "yap", "yelp", "yip", "yowl", "zoop", "zot", "biff", "boing", "brrr", "chug", "clash", "clatter", "coo", "crack", "creak", "dink", "dunk", "eek", "floop", "flump", "gong", "growl", "hic", "huff", "jangle", "kablooie", "kachunk", "kip", "klunk", "mush", "nark", "nyoom", "ooof", "paff", "pew", "phwoar", "plunk", "pow", "puff", "rump", "scrape", "shush", "skid", "slam", "sloop", "smoosh", "snarl", "snip", "sputter", "swipe", "tack", "tap", "thwomp", "tink", "tinkle", "vamp", "vwoop", "whump", "wizzle", "yee", "zizz", "blonk", "chaw", "dawp", "flizz", "gnash", "grrr", "hork", "jolt", "klink", "lump", "mow", "niff", "pang", "quonk", "riff", "scuff", "shlorp", "skronk", "snorf", "spang", "sproing", "thunk", "twerp", "vlop", "whizzle", "zibbit",
    "death", "gore", "splatter", "gush", "crack", "snap", "rip", "tear", "grind", "thrash", "slash", "stab", "bleed", "drip", "ooze", "slime", "puke", "choke", "gasp", "shriek", "howl", "scream", "wail", "moan", "groan", "rattle", "clank", "thud", "whump", "bash", "smash", "crush", "burst", "boom", "blast", "roar", "thunder", "fizz", "hiss", "sizzle", "pop", "spurt", "gloop", "squish", "squelch", "slosh", "plunge", "flop", "thwack", "whack", "smack", "slap", "punch", "claw", "gnash", "growl", "snarl", "yelp", "yowl", "squeal", "creak", "scrape", "grind", "rumble", "quake", "shatter", "splosh", "drown", "gurgle", "retch", "sputter", "froth", "bubble", "burst", "zap", "zing", "bzzzt", "vwoom", "whoosh", "vamp", "fang", "bite", "rip", "shred", "tear", "gash", "hack", "slice", "dice", "gut", "spill", "reek", "rot", "mold", "rust", "crumble", "dust", "ash", "smoke", "flame", "scorch", "blaze", "burn", "char", "flicker", "fade", "dim", "void", "black", "doom", "gloom", "crypt", "tomb", "grave", "skull", "bone", "ghost", "haunt", "shadow", "spook", "fear", "dread", "panic", "horror", "curse", "hex", "bane", "woe", "agony", "torment", "sting", "wound", "scar", "bruise", "blood", "plague", "toxin", "venom", "decay", "mire", "sludge", "mush", "flesh", "guts", "meat", "stink", "foul", "mourn", "weep", "sob", "whimper", "shiver", "quake", "tremble", "rattle", "clatter", "clang", "bang", "boom", "blast", "wreck", "ruin", "end"
]
NONSENSE_MEME = [
    "sus", "pog", "lmao", "nope", "yolo", "epic", "dank", "oofsize", "bigchungus", "stonks", "yeetus", "deletus", "vibe", "sadge", "cope", "mald", "cringe", "based", "redpilled", "sussy", "poggers", "kek", "bruhmoment", "sheesh", "bussin", "cap", "frfr", "lit", "slay", "fam", "goat", "hundo", "rizz", "salty", "simp", "thicc", "woke", "yas", "bet", "drip", "flex", "glowup", "jacked", "litfam", "noob", "pepe", "ratio", "savage", "snaccident", "tea", "whip", "yassify", "zoomer", "bloopers", "chad", "fomo", "gigachad", "hype", "karen", "lurker", "mood", "normie", "owned", "pwned", "reee", "shook", "stan", "troll", "uwu", "vibin", "wack", "yikes", "zesty", "clapback", "doggo", "edgy", "famalam", "giga", "heckin", "irl", "janky", "lowkey", "mid", "ngl", "opp", "preesh", "rando", "skrrt", "tiktok", "vaporwave", "whelp", "yeeted", "zonked", "banger", "boomer", "clout", "copium", "degen", "ez", "finesse", "gas", "glizzy", "highkey", "ish", "juju", "kino", "ligma", "mfw", "ngmi", "okboomer", "pilled", "qt", "rip", "sigma", "slaps", "sneed", "tfw", "vibecheck", "wagmi", "xqc", "yup", "zoinked", "amogus", "bait", "choke", "copypasta", "dilf", "eboys", "frick", "gfuel", "hitsdifferent", "iykyk", "jfc", "kys", "lfg", "milf", "nft", "omfg", "pogchamp", "qanon", "rawr", "simpin", "tposing", "unf", "vcard", "w", "xd", "yall", "zaddy", "bloopity", "chonkers", "dripcheck", "egirl", "femboy", "gamer", "heck", "inshallah", "jawn", "kpop", "lewd", "malding", "nfts", "oomf", "pogged", "qtpie", "ripped", "smd", "thot", "uwus", "vibey", "wokeaf", "xoxo", "yeehaw", "zooted",
    "death", "killer", "murder", "doom", "grave", "stab", "blood", "gore", "rot", "skull", "bone", "ghost", "haunt", "creep", "slay", "gloom", "drown", "choke", "bleed", "slash", "tomb", "crypt", "fear", "scream", "die", "kill", "pain", "void", "curse", "wound", "grim", "shadow", "spook", "thrash", "plague", "venom", "claw", "tear", "crush", "ash", "mourn", "rage", "horror", "panic", "dread", "evil", "hell", "zombie", "fang", "howl", "shriek", "gasp", "sob", "trap", "chain", "bury", "guts", "flesh", "reek", "mold", "rust", "crumble", "agony", "torment", "scar", "bruise", "lash", "shatter", "explode", "flame", "smoke", "strangle", "noose", "plunge", "sink", "fade", "black", "ice", "shiver", "quake", "blind", "lurk", "stalk", "hunt", "prey", "snare", "seize", "yank", "drag", "twitch", "spasm", "writhe", "flail", "pound", "batter", "ruin", "fate", "hex", "bane", "woe", "grief", "sting", "ache", "throb", "gush", "spurt", "drench", "swamp", "mire", "retch", "puke", "slobber", "froth", "burst", "thunder", "roar", "wail", "moan", "whine", "whimper", "tear", "plummet", "scorch", "blaze", "flicker", "dim", "cease", "stiff", "silent", "mute", "dead", "dust", "vapor", "mist", "missing", "hidden"
]
CUSS = [
    "damn", "hell", "shit", "fuck", "ass", "bitch", "crap", "piss" #add more
]
FILLERS = ["uhhhh", "ummm", "erm", "wellll", "sooooo", "uhh", "um", "er", "huh", "y’know", "ughhh","sooo", "prolly", "wellll", "sooooo", "uhhhhh", "ermm", "hmmmmmm", "um-uh", "er-uh", "uh-huh", "mm-hmm", "uhmm", "errrm", "well-uh", "so-uh", "kinda-uh", "y’know-uh", "basically-um", "like-uh", "sure-uh", "maybe-um", "dunno-uh", "hmmm-uh", "what-uh", "hey-um", "oh-uh", "gee-uh", "wait-um"]

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def fake_translate_mangle(text):
    return text  # Placeholder; remove or replace with argostranslate if needed

def prolong_last_letter(word):
    if random.random() < 0.05:  # 5% chance to prolong
        last_letter = word[-1]
        extra_letters = random.randint(2, 5)
        return word + last_letter * extra_letters
    return word

def alternate_caps(text):
    return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(text))

def get_synonym_or_antonym(word):
    """Get a random synonym or antonym from WordNet."""
    synsets = wordnet.synsets(word.lower())
    if not synsets:
        return word
    synonyms = set()
    antonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name().replace("_", " "))
    options = list(synonyms.union(antonyms))
    return random.choice(options) if options else word

def get_opposite_sentiment(word):
    """Get a word with opposite sentiment from VADER lexicon."""
    if word.lower() in sid.lexicon:
        score = sid.lexicon[word.lower()]
        # Find words with opposite sentiment (positive -> negative, negative -> positive)
        opposites = [w for w, s in sid.lexicon.items() if (s < 0 if score > 0 else s > 0) and w != word.lower()]
        return random.choice(opposites) if opposites else word
    return word

def mangle_text(text):
    """Mangle non-tag text with synonym/antonym swaps, sentiment flips, capitals, cussing, and fillers."""
    parts = re.split(r"(\{\{.*?\}\})", text)
    new_parts = []
    for part in parts:
        if part.startswith("{{") and part.endswith("}}"):
            new_parts.append(part)  # Preserve tags
        else:
            part = fake_translate_mangle(part)
            words = part.split()
            if not words:
                new_parts.append(part)
                continue
            new_words = []
            for word in words:
                roll = random.random()
                if roll < 0.4:  # 30% chance for synonym/antonym
                    new_words.append(prolong_last_letter(get_synonym_or_antonym(word)))
                elif roll < 0.6:  # 20% chance for opposite sentiment (30% to 50% range)
                    new_words.append(prolong_last_letter(get_opposite_sentiment(word)))
                else:
                    new_words.append(prolong_last_letter(word))
                # 5% cuss injection
                if random.random() < 0.05:
                    cuss = random.choice(CUSS)
                    cuss = prolong_last_letter(cuss)
                    cuss = cuss.lower() if random.random() < 0.5 else cuss.upper()
                    new_words.append(cuss)

            # 10% filler word injection
            if random.random() < 0.1:
                new_words.insert(random.randint(0, len(new_words)), random.choice(FILLERS))

            # 5% alternate caps
            mangled_text = " ".join(new_words)
            if random.random() < 0.05:
                mangled_text = alternate_caps(mangled_text)

            new_parts.append(mangled_text)
    return "".join(new_parts)

def process_section(section):
    """Process a section, mangling only text lines."""
    lines = section.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith(("label:", "attribute:")):
            new_lines.append(line)
        else:
            new_lines.append(mangle_text(line))
    return "\n".join(new_lines)

def process_section(section):
    """Process a section, mangling only text lines."""
    lines = section.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith(("label:", "attribute:")):
            new_lines.append(line)
        else:
            new_lines.append(mangle_text(line))
    return "\n".join(new_lines)

def process_section(section):
    """Process a section, mangling only text lines."""
    lines = section.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith("label:") or line.strip().startswith("attribute:"):
            new_lines.append(line)
        else:
            new_lines.append(mangle_text(line))
    return "\n".join(new_lines)

def detect_encoding(file_path):
    """Detect file encoding by checking byte signature."""
    with open(file_path, "rb") as f:
        raw_bytes = f.read(4)  # Read first 4 bytes for BOM check
    if raw_bytes.startswith(b'\xFF\xFE'):  # UTF-16-LE BOM
        return "utf-16-le"
    elif raw_bytes.startswith(b'\xFE\xFF'):  # UTF-16-BE BOM
        return "utf-16-be"
    elif raw_bytes.startswith(b'\xEF\xBB\xBF'):  # UTF-8 BOM
        return "utf-8-sig"
    else:
        # No BOM, guess based on common encodings (try utf-16-le first for ACNH)
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
                return "latin-1"  # Fallback

# Get input and output folders from user
input_root = input("Enter the input folder path (e.g., C:/path/to/unmodded): ").strip()
output_root = input("Enter the output folder path (e.g., C:/path/to/modded): ").strip()

# Ensure output root exists
if not os.path.exists(output_root):
    os.makedirs(output_root)

# Count total .txt files for progress tracking
total_files = sum(1 for root, _, files in os.walk(input_root) for file in files if file.endswith(".txt"))
current_file = 0

# Process files, preserving folder structure
for root, dirs, files in os.walk(input_root):
    relative_path = os.path.relpath(root, input_root)
    output_subfolder = os.path.join(output_root, relative_path)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    for file in files:
        if file.endswith(".txt"):
            current_file += 1
            input_file_path = os.path.join(root, file)
            output_file_path = os.path.join(output_subfolder, file)
            print(f"Processing {current_file}/{total_files}: {file} from {root} to {output_subfolder}...")

            # Detect encoding
            encoding = detect_encoding(input_file_path)
            print(f"Detected encoding for {file}: {encoding}")

            # Read with detected encoding
            try:
                with open(input_file_path, "r", encoding=encoding) as f:
                    raw_text = f.read()
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                continue

            # Split header and content
            parts = raw_text.split("---", 1)
            header = parts[0].strip()
            if len(parts) <= 1:
                modded_text = header
            else:
                sections = parts[1].strip().split("\n---\n")
                mangled_sections = [process_section(section) for section in sections]
                modded_text = f"{header}\n---\n" + "\n---\n".join(mangled_sections)

            # Write with the same encoding as input
            try:
                with open(output_file_path, "w", encoding=encoding) as f:
                    f.write(modded_text)
                print(f"Wrote {current_file}/{total_files}: {file} to {output_file_path}")
            except Exception as e:
                print(f"Failed to write {file}: {e}")

print("Done! Convert .txt files from the output folder back to .msbt.")