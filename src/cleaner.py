import re
from difflib import get_close_matches

# Applied character-by-character BEFORE uppercasing.
# Each maps a consistent EasyOCR misread of the Exocet font:
#   e  -> O   narrow oval O is read as lowercase e  (SweRD, Te, Beets)
#   @  -> O   alternate O misread                   (SANDST@RM, @F)
#   $  -> S                                          (WOUND$, SEceND$)
#   [  -> 1   serif digit 1 looks like bracket       (+[, [2 of [2)
#   €  -> E   Exocet E sometimes read as euro sign   (DAMAG €)
_PRE_TRANS = str.maketrans({'e': 'O', '@': 'O', '$': 'S', '[': '1', '€': 'E'})

# D2R vocabulary used for fragment merging and fuzzy word correction.
_VOCAB = frozenset({
    # Stats / properties
    "DEFENSE", "DAMAGE", "DURABILITY", "REQUIRED", "STRENGTH", "DEXTERITY",
    "VITALITY", "ENERGY", "LEVEL", "CHANCE", "FASTER", "INCREASED",
    "ENHANCED", "MAXIMUM", "MINIMUM", "RESIST", "RESISTANCE", "ABSORB",
    "REPLENISH", "REGENERATE", "STAMINA", "MISSILE", "RECOVERY", "BLOCK",
    "ATTACK", "CAST", "RATE", "SPEED", "STRIKE", "STRIKING", "CHAIN",
    "STATIC", "FIELD", "IGNORE", "TARGET", "ENEMY", "OPEN", "WOUND",
    "BONUS", "INVENTORY", "SOCKETED", "ETHEREAL", "REPAIRED", "CANNOT",
    "SKILLS", "SKILL", "MANA", "LIFE", "GOLD", "MONSTERS", "EXTRA",
    "BETTER", "ITEMS", "GETTING", "ATTRIBUTES", "CHARGES", "SUMMON",
    "PORTAL", "SCROLL", "SCROLLS", "QUANTITY", "SLOWS", "SLOWER",
    "DRAIN", "REPAIRS", "SECONDS", "SECOND", "INDESTRUCTIBLE",
    "POISON", "COLD", "FIRE", "LIGHTNING", "MAGIC", "PHYSICAL",
    "ALL", "TO", "OF", "ON", "IN", "BY", "AND", "VS",
    "BASED", "CHARACTER", "EACH", "AFTER", "KILL", "ATTACKER", "TAKES",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "HAND", "HANDED",
    "CLASS", "FAST", "NORMAL", "HIT", "ADDS", "ITEM",
    "INCREASE", "EXPERIENCE", "GAINED", "RADIUS", "LIGHT",
    # Item types
    "SWORD", "AXE", "MACE", "FLAIL", "SCEPTER", "BOW", "WAND", "STAFF",
    "ORB", "DAGGER", "BLADE", "FALCHION", "SCIMITAR", "SHIELD", "BUCKLER",
    "HELM", "CAP", "MASK", "SKULL", "CROWN", "DIADEM", "TIARA",
    "ARMOR", "PLATE", "MAIL", "GLOVES", "GAUNTLETS", "BOOTS", "GREAVES",
    "BELT", "GIRDLE", "SASH", "RING", "AMULET", "CHARM", "SMALL",
    "LARGE", "GRAND", "RUNE", "GEM", "JEWEL", "SPIDERWEB", "SCARABSHELL",
    "CRYSTAL", "MONARCH",
    # Rune names (longer ones only — short rune names are too risky to fuzzy-correct)
    "SHAEL",
    # Unique / runeword names
    "CRESCENT", "MOON", "GRIFFON", "SANDSTORM", "TREK", "ARACHNID",
    "MESH", "HAVOC", "HELLFIRE", "TORCH", "SPARKING", "CRACK",
    "HEAVENS", "GRIM", "TALISMAN", "SPIRIT", "EYE", "TOWN", "TOME",
    # Skills / procs
    "CYCLONE", "FIRESTORM", "HYDRA", "VENOM", "CHAIN", "NOVA",
    # Class names
    "SORCERESS", "PALADIN", "BARBARIAN", "NECROMANCER", "AMAZON",
    "DRUID", "ASSASSIN",
    # Other tooltip words
    "ONLY", "SPECIFIC", "INSERT", "CLICK", "RIGHT", "USE",
    "SIZE", "SLOTS", "KEY", "KEEP", "GAIN", "MONSTER",
    "SUNDERED", "SUNDERING", "IMMUNITY", "RESISTANCES",
})

# Single-word item-type tokens that mark the END of an item name when they appear
# as a standalone line (not the very first line of the tooltip).
_ITEM_TYPE_WORDS = frozenset({
    "SWORD", "AXE", "MACE", "FLAIL", "SCEPTER", "BOW", "WAND", "STAFF",
    "ORB", "DAGGER", "BLADE", "FALCHION", "SCIMITAR", "SHIELD", "BUCKLER",
    "HELM", "CAP", "MASK", "SKULL", "CROWN", "DIADEM", "TIARA",
    "ARMOR", "PLATE", "MAIL", "GLOVES", "GAUNTLETS", "BOOTS", "GREAVES",
    "BELT", "GIRDLE", "SASH", "RING", "AMULET",
    "CHARM", "LARGE", "SMALL", "GRAND",
    "MONARCH",
})


def clean_line(line: str) -> str:
    # 1. Pre-uppercase character substitutions
    line = line.translate(_PRE_TRANS)
    # 2. Uppercase everything
    line = line.upper()
    # 3. Replace digit-zero directly adjacent to a letter → letter O
    #    "T0" → "TO",  "0F" → "OF",  but "30%" stays "30%"
    line = re.sub(r'(?<=[A-Z])0|0(?=[A-Z])', 'O', line)
    # 4. Collapse spurious spaces inserted within digit sequences
    #    Must run before step 5 so "8 0 TO" → "80 TO" before the 0 is converted
    for _ in range(4):
        line = re.sub(r'(\d) (\d)', r'\1\2', line)
    # 5. Replace digit-zero separated by a single space from a letter → letter O
    #    "0 F" → "O F" (fragment merge in step 6 then joins to "OF")
    #    Guard: don't touch zero that follows another digit ("20 FASTER" safe)
    line = re.sub(r'(?<![0-9])0 (?=[A-Z])', 'O ', line)
    # 6. Merge consecutive word-fragments that together form a vocab word
    line = _merge_fragments(line)
    # 7. Fuzzy per-word correction against D2R vocabulary
    tokens = line.split()
    tokens = [_correct_word(t) for t in tokens]
    return ' '.join(tokens).strip()


def _merge_fragments(line: str) -> str:
    """Iteratively merge adjacent tokens that together spell a vocab word."""
    words = line.split()
    changed = True
    while changed:
        changed = False
        result: list[str] = []
        i = 0
        while i < len(words):
            # Try three-token merge first (e.g. "M O NSTERS" -> "MONSTERS")
            if i + 2 < len(words):
                m3 = words[i] + words[i + 1] + words[i + 2]
                if (m3 in _VOCAB
                        and words[i] not in _VOCAB
                        and words[i + 1] not in _VOCAB
                        and words[i + 2] not in _VOCAB):
                    result.append(m3)
                    i += 3
                    changed = True
                    continue
            # Try two-token merge (e.g. "INVENTO" + "RY" -> "INVENTORY")
            if i + 1 < len(words):
                m2 = words[i] + words[i + 1]
                if (m2 in _VOCAB
                        and words[i] not in _VOCAB
                        and words[i + 1] not in _VOCAB):
                    result.append(m2)
                    i += 2
                    changed = True
                    continue
            result.append(words[i])
            i += 1
        words = result
    return ' '.join(words)


def _correct_word(word: str) -> str:
    """Fuzzy-correct a single uppercase token against the D2R vocabulary."""
    if not word.isalpha() or len(word) < 4:
        return word
    if word in _VOCAB:
        return word
    matches = get_close_matches(word, _VOCAB, n=1, cutoff=0.75)
    return matches[0] if matches else word


def _is_stat_line(line: str) -> bool:
    """Return True for lines that are clearly stat values rather than item name words."""
    return (
        ':' in line
        or line.startswith('+')
        or line.startswith('-')
        or line[:1].isdigit()
        or (line.endswith('%') and any(c.isdigit() for c in line))
    )


def _ends_title(line: str, is_first: bool) -> bool:
    """
    Return True if this line should stop title accumulation.
    Stat lines always stop. On non-first lines, a line whose last word is a
    known item-type word (e.g. "CRYSTAL SWORD", "DIADEM", "LARGE") also stops.
    """
    if _is_stat_line(line):
        return True
    if not is_first:
        last = line.split()[-1] if line else ''
        if last in _ITEM_TYPE_WORDS:
            return True
    return False


def structure_tooltip(lines: list[str]) -> dict:
    """
    Split cleaned lines into {item_title, item_details}.

    item_title  — leading non-stat, non-type lines joined into one string.
                  Capped at 4 lines to handle multi-word names like
                  "TOME OF TOWN PORTAL" without over-consuming detail lines.
    item_details — remaining lines as a list (one cleaned line per element).
    """
    if not lines:
        return {"item_title": "", "item_details": []}

    title_parts: list[str] = []
    for line in lines[:4]:
        if _ends_title(line, is_first=(len(title_parts) == 0)):
            break
        title_parts.append(line)

    if not title_parts:
        title_parts = [lines[0]]

    title = ' '.join(title_parts)
    details = lines[len(title_parts):]
    return {"item_title": title, "item_details": details}


def clean_and_structure(lines: list[str]) -> dict:
    """Clean raw OCR lines and return structured {item_title, item_details}."""
    cleaned = [cl for line in lines if (cl := clean_line(line))]
    return structure_tooltip(cleaned)


if __name__ == "__main__":
    import json, sys, pathlib
    src = sys.argv[1] if len(sys.argv) > 1 else "items.json"
    dst = str(pathlib.Path(src).with_stem(pathlib.Path(src).stem + "_cleaned"))
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    out = [clean_and_structure(r["lines"]) for r in data]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Cleaned {len(out)} entries -> {dst}")
