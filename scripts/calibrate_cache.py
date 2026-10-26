"""Pick SEMANTIC_CACHE_THRESHOLD from labelled question pairs.

    docker compose exec api python -m scripts.calibrate_cache

"same" pairs should share a cached answer; "different" pairs look alike but need different
answers (different product, term, fee or audience). e5 similarities run high even for unrelated
text, so the threshold has to sit above every "different" pair: a false hit serves a wrong
answer, while a miss only costs one LLM call. The cache also requires matching numbers
(numbers_in), which is applied here too.
"""

import numpy as np

from app.services.embeddings import get_embedder
from app.services.semantic_cache import numbers_in

SAME = [
    ("What is the late payment fee on the credit card?", "What's the credit card late payment fee?"),
    ("How long are reward points valid?", "How long do reward points stay valid?"),
    ("How long are reward points valid?", "What is the validity of reward points?"),
    ("FD rate for senior citizens for 1 year", "senior citizen fixed deposit interest rate 1 year"),
    ("Do I earn points on fuel purchases?", "Do fuel purchases earn reward points?"),
    ("points kab expire hote hain", "points kab expire hote hai"),
    ("card kho gaya toh kya kare", "card kho jaye toh kya karna chahiye"),
    ("पॉइंट्स कब खत्म होते हैं", "पॉइंट्स कब समाप्त होते हैं"),
    ("What is the annual fee for the credit card?", "Credit card annual fee kitni hai?"),
]
DIFFERENT = [
    ("FD rate for senior citizens for 1 year", "FD rate for senior citizens for 2 years"),
    ("FD rate for senior citizens for 1 year", "FD rate for 1 year"),
    ("What is the late payment fee on the credit card?", "What is the annual fee on the credit card?"),
    ("How long are reward points valid?", "How can reward points be redeemed?"),
    ("Do I earn points on fuel purchases?", "Do I earn points on rent payments?"),
    ("points kab expire hote hain", "points kaise redeem karte hain"),
    ("card kho gaya toh kya kare", "card block kaise kare"),
    ("What is the cash withdrawal fee at ATMs?", "What is the foreign currency markup fee?"),
    ("home loan interest rate", "car loan interest rate"),
]


def _similarities(pairs: list[tuple[str, str]]) -> list[float]:
    embedder = get_embedder()
    a = embedder.embed_queries([p[0] for p in pairs])
    b = embedder.embed_queries([p[1] for p in pairs])
    return [float(x) for x in np.sum(a * b, axis=1)]  # unit vectors: dot product == cosine


def main() -> None:
    same, different = _similarities(SAME), _similarities(DIFFERENT)
    for label, pairs, sims in (("same", SAME, same), ("different", DIFFERENT, different)):
        print(f"\n{label}")
        for (q1, q2), sim in sorted(zip(pairs, sims, strict=True), key=lambda x: -x[1]):
            print(f"  {sim:.4f}  {q1!r} ~ {q2!r}")

    def guarded(pairs, sims):
        return [s if numbers_in(q1) == numbers_in(q2) else 0.0 for (q1, q2), s in zip(pairs, sims, strict=True)]

    print(f"\nhighest 'different' pair: {max(different):.4f}")
    for name, same_s, diff_s in (
        ("similarity only", same, different),
        ("+ number guard", guarded(SAME, same), guarded(DIFFERENT, different)),
    ):
        print(f"\n{name}")
        for t in (0.93, 0.95, 0.96, 0.965, 0.97):
            hits = sum(s >= t for s in same_s)
            false_hits = sum(s >= t for s in diff_s)
            print(f"  threshold {t:.3f}: {hits}/{len(SAME)} paraphrases hit, {false_hits}/{len(DIFFERENT)} false hits")


if __name__ == "__main__":
    main()
