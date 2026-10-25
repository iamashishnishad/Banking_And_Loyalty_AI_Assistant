"""Language detection, Devanagari -> Roman transliteration, and phonetic keys for Hinglish.

Hinglish has no standard spelling: the same word shows up as "khatam", "khatm" and "खत्म", or
"card" and "कार्ड". Search handles this by indexing a *phonetic key* per word, built from a
casual romanisation, so all spellings of a word collapse to one key:

    "खत्म" -> "khatm" -> "ktm"    "khatam" -> "ktm"
    "कार्ड" -> "kaard" -> "krd"    "card"   -> "krd"
"""

import re
from typing import Literal

Language = Literal["english", "hindi", "hinglish"]

_DEVANAGARI_RE = re.compile(f"[{chr(0x0900)}-{chr(0x097F)}]")
_LATIN_WORD_RE = re.compile(r"[a-z]+")

# Common romanised Hindi words. Words that are also everyday English ("me", "to", "the", "do",
# "band", "mat", "din", "main") are left out so English sentences aren't misread as Hinglish.
HINGLISH_LEXICON: frozenset[str] = frozenset(
    """
    hai hain tha thi hoga hogi honge hota hoti hote ho raha rahi rahe kya kyu kyun kyon kaise
    kaisa kaisi kab kahan kaha kitna kitne kitni kaun koi kuch bhi nahi nahin haan ji accha
    acha theek thik mera meri mere mujhe mujhko humara hamara hamare apna apni apne aap aapka
    aapki aapke tum tumhara tera teri unka uska iska yeh ye woh wo vo aur ya lekin par pe ke ki ka
    ko se mein liye wala wali wale chahiye chahta chahti sakta sakti sakte karna karo karein kare
    karu karoon kardo kar diya diye dena dedo lena milega milta milte milenge milegi mila
    mili gaya gayi gaye jata jati jaate jayega jaayega hua hui hue bata batao bataiye batana btao
    paisa paise rupaye kab tak abhi jaldi pehle baad mahina mahine saal hafte chalu
    khatam khatm kho bhool galat sahi dobara wapas zyada jyada kam bahut sabse kyunki agar toh
    """.split()
)


def detect_language(text: str) -> Language:
    """'hindi' if the message contains Devanagari words, 'hinglish' if its Latin words are largely
    romanised Hindi, else 'english'. Mixed script ("मेरा credit card block करो") is Hindi: people who
    type Devanagari at all expect Devanagari back."""
    if len(_DEVANAGARI_RE.findall(text)) >= 3:
        return "hindi"
    words = _LATIN_WORD_RE.findall(text.lower())
    if not words:
        return "english"
    hits = sum(w in HINGLISH_LEXICON for w in words)
    if hits >= 2 or hits / len(words) > 0.34:
        return "hinglish"
    return "english"


# --- Devanagari -> casual Roman -------------------------------------------------------------

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
# Consonant + nukta (NFKC decomposes क़, ज़, फ़ ... into base + U+093C).
_NUKTA_FORMS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
_NUKTA = chr(0x093C)
_VIRAMA = chr(0x094D)
_NASALS = {chr(0x0902): "n", chr(0x0901): "n"}  # anusvara, chandrabindu
_VISARGA = chr(0x0903)
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def transliterate(word: str) -> str:
    """Romanise Devanagari the way people type Hinglish: no diacritics, inherent 'a' kept between
    consonants but dropped at the end of a word (राम -> raam, not raama). Non-Devanagari characters
    pass through. Approximate by design: it only has to agree with itself and with common spellings
    after phonetic_key()."""
    out: list[str] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _CONSONANTS:
            roman = _CONSONANTS[ch]
            if i + 1 < len(chars) and chars[i + 1] == _NUKTA:
                roman = _NUKTA_FORMS.get(ch, roman)
                i += 1
            out.append(roman)
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            # Inherent vowel only when another consonant follows directly (no matra/virama).
            if nxt in _CONSONANTS:
                out.append("a")
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
        elif ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
        elif ch in _NASALS:
            out.append(_NASALS[ch])
        elif ch == _VISARGA:
            out.append("h")
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch in (_VIRAMA, _NUKTA):
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# --- Phonetic key -----------------------------------------------------------------------------

# Applied in order. "ch" is parked on a placeholder so that the English c -> k rule doesn't merge
# "chai" into "kai".
_PHONETIC_RULES: list[tuple[str, str]] = [
    ("chh", "%"), ("ch", "%"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"),
    ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"),
    ("z", "j"), ("c", "k"), ("%", "c"),
]
_VOWELS = set("aeiou")


def phonetic_key(word: str) -> str:
    """Consonant skeleton of a romanised word: aspirates and c/k/q, w/v, z/j merged, vowels dropped
    (except a leading one), doubled letters collapsed. Returns '' for words too short to key."""
    w = transliterate(word).lower()
    if not w.isascii() or not w.isalpha():
        return ""
    for src, dst in _PHONETIC_RULES:
        w = w.replace(src, dst)
    head, tail = w[0], w[1:]
    tail = "".join(ch for ch in tail if ch not in _VOWELS and ch != "h")
    key = head + tail
    key = re.sub(r"(.)\1+", r"\1", key)
    return key if len(key) >= 2 else ""
