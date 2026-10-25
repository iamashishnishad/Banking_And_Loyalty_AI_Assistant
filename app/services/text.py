"""Text normalisation shared by indexing and querying.

The lexical index uses Postgres' 'simple' text-search config (no stemming, no stopwords), because
none of the built-in configs handle Hindi or romanised Hinglish. Stopword removal and spelling
normalisation therefore happen here, and the exact same function runs on documents and queries.
"""

import re
import unicodedata

from app.services.language import phonetic_key

# Letters/digits (no underscore) plus the whole Devanagari block: \w alone misses vowel signs and
# viramas (category M), which would split words like "पॉइंट्स" into fragments.
_DEVANAGARI = f"{chr(0x0900)}-{chr(0x0963)}{chr(0x0966)}-{chr(0x097F)}"  # U+0900-097F minus danda । ॥
_TOKEN_RE = re.compile(rf"(?:[^\W_]|[{_DEVANAGARI}])+")

# Common English, romanised Hindi and Devanagari function words. They appear in nearly every
# question ("mere kitne points hai") and, without IDF weighting, would drown out the useful terms.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the is are was were be been am do does did to of in on at for from by with and or but
    if then so as it its this that these those i me my we our you your he she they them their
    what which who whom how when where why can could will would should shall may might must
    please tell about any some there here have has had not no yes also just than too very
    hai hain ho hoga hogi hote hota hoti tha thi the ka ki ke ko se me mein mai main mera meri
    mere hum humara apna apne apni aap aapka aapki aapke tum tumhara kya kyu kyun kaise kaisa
    kab kahan kaha kitna kitne kitni koi kuch bhi to toh na nahi nahin haan ji bata batao
    batana btao bataye bataiye karo karna karein kar raha rahi rahe wala wali wale yeh ye woh
    wo vo is us iska uska aur ya par pe liye lie
    है हैं हो था थी थे का की के को से में मैं मेरा मेरी मेरे हम आप आपका आपकी आपके क्या कैसे
    कब कहाँ कितना कितने कितनी कोई कुछ भी तो न नहीं हाँ जी बताओ बताइए करें करना और या पर लिए
    यह ये वह वो इस उस
    """.split()
)

# Spelling variants seen in romanised Hinglish, mapped to one canonical form so documents and
# queries agree. Phase 2 replaces this with proper transliteration.
SPELLING_VARIANTS: dict[str, str] = {
    "pts": "points",
    "point": "points",
    "pointz": "points",
    "reward": "rewards",
    "redemption": "redeem",
    "redeemed": "redeem",
    "redeeming": "redeem",
    "expiry": "expire",
    "expires": "expire",
    "expired": "expire",
    "expiring": "expire",
    "khatam": "expire",
    "cc": "credit",
    "paisa": "paise",
    "rupaye": "rupees",
    "rupay": "rupees",
    "rs": "rupees",
    "inr": "rupees",
    # Common Devanagari banking terms mapped to the English term used in most policy documents.
    "पॉइंट्स": "points",
    "पॉइंट": "points",
    "अंक": "points",
    "रिवॉर्ड": "rewards",
    "खत्म": "expire",
    "समाप्त": "expire",
    "रिडीम": "redeem",
    "कार्ड": "card",
    "क्रेडिट": "credit",
    "डेबिट": "debit",
    "ब्याज": "interest",
    "शुल्क": "fee",
    "फीस": "fee",
}


def normalize(text: str) -> str:
    """Unicode-normalise and collapse whitespace. Used for the stored chunk content."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(normalize(text).lower()):
        token = SPELLING_VARIANTS.get(raw, raw)
        if token in STOPWORDS or (len(token) == 1 and token.isascii()):
            continue
        tokens.append(token)
    return tokens


def lexical_text(text: str) -> str:
    """Space-joined tokens stored in chunks.lexical_text and indexed as a tsvector."""
    return " ".join(tokenize(text))


def to_or_tsquery(text: str) -> str | None:
    """Build an OR tsquery string ('a | b | c'). OR keeps recall high; ts_rank_cd still ranks
    chunks that match more terms higher. Tokens are letters/digits only, so they cannot inject tsquery syntax.
    Returns None when nothing searchable is left (e.g. a query made only of stopwords)."""
    terms = list(dict.fromkeys(tokenize(text)))
    if not terms:
        return None
    return " | ".join(terms)


def phonetic_text(text: str) -> str:
    """Space-joined phonetic keys (see language.phonetic_key) for chunks.phonetic_text. Lets
    'khatam', 'khatm' and 'खत्म' match each other. Pure numbers are left to the lexical index."""
    keys = (phonetic_key(t) for t in tokenize(text) if not t.isdigit())
    return " ".join(k for k in keys if k)


def to_phonetic_tsquery(text: str) -> str | None:
    keys = list(dict.fromkeys(phonetic_text(text).split()))
    return " | ".join(keys) if keys else None
