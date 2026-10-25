import unicodedata

import pytest

from app.services.language import detect_language, phonetic_key, transliterate
from app.services.text import phonetic_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mere points kab expire honge", "hinglish"),
        ("card kho gaya", "hinglish"),
        ("balance batao", "hinglish"),
        ("What is the FD rate?", "english"),
        ("Is the fee waived?", "english"),
        ("Do I get points on fuel?", "english"),
        ("मेरे पॉइंट्स कब खत्म होंगे", "hindi"),
        ("मेरा credit card block करो", "hindi"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


@pytest.mark.parametrize(
    ("devanagari", "roman"),
    [("खत्म", "khatm"), ("कार्ड", "kaard"), ("पॉइंट्स", "points"), ("राम", "raam"), ("होंगे", "honge")],
)
def test_transliterate(devanagari, roman):
    assert transliterate(devanagari) == roman


def test_nukta_forms():
    # NFKC decomposes precomposed nukta letters; both forms must romanise the same.
    assert transliterate(unicodedata.normalize("NFKC", "ज़्यादा")).startswith("z")


@pytest.mark.parametrize(
    "variants",
    [("khatam", "khatm", "खत्म"), ("card", "कार्ड"), ("zyada", "jyada", "ज़्यादा"), ("block", "ब्लॉक"), ("fees", "फ़ीस")],
)
def test_spelling_variants_share_a_phonetic_key(variants):
    keys = {phonetic_key(unicodedata.normalize("NFKC", v)) for v in variants}
    assert len(keys) == 1 and "" not in keys


def test_phonetic_text_crosses_scripts():
    assert phonetic_text("मेरे पॉइंट्स कब खत्म होंगे") == phonetic_text("mere points kab khatam honge")
