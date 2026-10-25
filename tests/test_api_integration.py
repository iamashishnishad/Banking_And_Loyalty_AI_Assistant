"""End-to-end tests against a running stack with seed data loaded:

    docker compose up -d && docker compose exec api python -m scripts.seed
    pytest tests/test_api_integration.py

Skipped automatically when the API isn't reachable.
"""

import io
import os
import uuid

import httpx
import pytest
from docx import Document as DocxDocument

API_URL = os.getenv("API_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def client():
    try:
        httpx.get(f"{API_URL}/health/ready", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"API not reachable at {API_URL}")
    with httpx.Client(base_url=API_URL, timeout=60) as c:
        yield c


def _search(client, query, tenant="demo-bank", **kw):
    r = client.post("/search", json={"query": query, **kw}, headers={"X-Tenant": tenant})
    r.raise_for_status()
    return r.json()


@pytest.mark.parametrize(
    ("query", "expected_doc_fragment"),
    [
        ("How long are reward points valid?", "Rewards"),
        ("mere points kab expire honge", "Rewards"),
        ("मेरे पॉइंट्स कब खत्म होंगे", "Rewards"),
        ("card kho gaya kya karu", "डेबिट"),
        ("senior citizen FD rate 2 years", "Fixed Deposit"),
        ("late payment fee kitni hai", "Fees"),
    ],
)
def test_multilingual_queries_hit_the_right_document(client, query, expected_doc_fragment):
    hits = _search(client, query, top_k=3)["hits"]
    assert hits, query
    assert any(expected_doc_fragment.lower() in h["document_title"].lower() for h in hits), [
        h["document_title"] for h in hits
    ]


def test_hybrid_reports_both_retrievers(client):
    hits = _search(client, "reward points expire 36 months")["hits"]
    assert any(h["vector_rank"] and h["lexical_rank"] for h in hits)


def test_stopword_only_query_falls_back_to_vector(client):
    body = _search(client, "kya hai", mode="hybrid")
    assert all(h["lexical_rank"] is None for h in body["hits"])


def test_tenant_isolation(client):
    for hit in _search(client, "Second Bank points never expire", top_k=20)["hits"]:
        assert "second bank" not in hit["document_title"].lower()
    titles = [h["document_title"] for h in _search(client, "points expire", tenant="second-bank")["hits"]]
    assert titles and all("Second Bank" in t for t in titles)


def test_unknown_tenant_404(client):
    r = client.post("/search", json={"query": "x"}, headers={"X-Tenant": "no-such-bank"})
    assert r.status_code == 404


def test_docx_upload_is_searchable_and_deduplicated(client):
    marker = f"zebracode{uuid.uuid4().hex[:8]}"
    doc = DocxDocument()
    doc.add_paragraph(f"Lounge access policy {marker}: Signature cardholders get 8 free airport lounge visits a year.")
    buf = io.BytesIO()
    doc.save(buf)
    files = {"file": ("lounge_policy.docx", buf.getvalue())}

    first = client.post("/documents", files=files)
    assert first.status_code == 201 and first.json()["created"] is True
    second = client.post("/documents", files=files)
    assert second.json()["created"] is False
    doc_id = first.json()["document"]["id"]

    hits = _search(client, marker, mode="lexical")["hits"]
    assert hits and hits[0]["document_id"] == doc_id

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert not _search(client, marker, mode="lexical")["hits"]


def test_unsupported_upload_rejected(client):
    r = client.post("/documents", files={"file": ("x.png", b"\x89PNG....")})
    assert r.status_code == 415


def test_balance_and_idempotent_redeem(client):
    before = client.get("/customers/CUST1002/points").json()["points"]
    ref = f"test-{uuid.uuid4()}"
    r = client.post("/customers/CUST1002/redeem", json={"points": 100, "reference": ref})
    assert r.status_code == 200 and r.json()["new_balance"] == before - 100
    dup = client.post("/customers/CUST1002/redeem", json={"points": 100, "reference": ref})
    assert dup.status_code == 409
    assert client.get("/customers/CUST1002/points").json()["points"] == before - 100


def test_redeem_more_than_balance_rejected(client):
    r = client.post("/customers/CUST1003/redeem", json={"points": 999_999, "reference": f"t-{uuid.uuid4()}"})
    assert r.status_code == 409


def test_chat_balance_hinglish(client):
    r = client.post("/chat", json={"customer_ref": "CUST1001", "message": "mere kitne points hai?"})
    body = r.json()
    assert body["intent"] == "points_balance"
    assert "points hain" in body["answer"]
    assert "expire" in body["answer"]  # CUST1001 has 1800 points expiring within 30 days


def test_chat_faq_returns_sources(client):
    body = client.post("/chat", json={"customer_ref": "CUST1001", "message": "Fuel par points milte hain kya?"}).json()
    assert body["intent"] == "faq" and body["sources"]
