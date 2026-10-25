from app.services.text import lexical_text, to_or_tsquery, tokenize


def test_hinglish_stopwords_removed_and_variants_normalised():
    assert tokenize("Mere reward pts kab expire honge?") == ["rewards", "points", "expire", "honge"]


def test_devanagari_words_stay_whole_and_map_to_english_terms():
    tokens = tokenize("मेरे पॉइंट्स कब खत्म होंगे।")
    assert tokens == ["points", "expire", "होंगे"]  # danda stripped, stopwords मेरे/कब dropped


def test_or_tsquery_dedupes_terms():
    assert to_or_tsquery("points points kitne points") == "points"
    assert to_or_tsquery("FD rate for senior citizens") == "fd | rate | senior | citizens"


def test_query_of_only_stopwords_gives_no_tsquery():
    assert to_or_tsquery("kya hai?") is None


def test_tsquery_operators_cannot_be_injected():
    assert to_or_tsquery("points & !fee | (x) <-> y:*") == "points | fee"  # single letters are dropped


def test_lexical_text_matches_query_tokens():
    assert lexical_text("Reward Points expire after 36 months") == "rewards points expire after 36 months"
