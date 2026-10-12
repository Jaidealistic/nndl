"""BLACK-BOX TESTS - Flask API / UI contract (Jai).
Techniques: equivalence partitioning, boundary-value analysis, error-guessing,
negative testing, response-contract (schema) testing. No knowledge of internals used."""
import io
import os

import pytest

from conftest import SAMPLES

VERDICTS = {"Genuine", "Font Tampered", "Metadata Tampered", "Both Tampered"}
RESULT_KEYS = {"verdict", "confidence", "probabilities", "metadata_inspection",
               "forensic_explanation", "heatmap_base64", "original_base64", "filename"}


def _post(client, data, name="doc.pdf", content_type="application/pdf"):
    return client.post("/predict", data={"file": (io.BytesIO(data), name, content_type)},
                       content_type="multipart/form-data")


# ---------- GET endpoints ----------
def test_health_contract(client):
    b = client.get("/health").get_json()
    assert {"status", "model", "device", "classes"} <= set(b)
    assert len(b["classes"]) == 4


def test_samples_catalog_lists_four_samples(client):
    s = client.get("/samples").get_json()["samples"]
    assert len(s) == 4
    assert {x["expected"] for x in s} == VERDICTS


def test_every_catalog_sample_file_exists(client):
    from conftest import DEMO_DIR
    for s in client.get("/samples").get_json()["samples"]:
        assert os.path.isfile(os.path.join(DEMO_DIR, s["id"])), s["id"]


@pytest.mark.parametrize("key", list(SAMPLES))
def test_predict_sample_response_schema(client, key):
    r = client.get(f"/predict_sample/{SAMPLES[key]}")
    assert r.status_code == 200
    b = r.get_json()
    assert RESULT_KEYS <= set(b)
    assert b["verdict"] in VERDICTS
    assert set(b["probabilities"]) == VERDICTS
    assert len(b["metadata_inspection"]) == 13


def test_predict_sample_unknown_returns_404(client):
    assert client.get("/predict_sample/nope.pdf").status_code == 404


@pytest.mark.parametrize("evil", ["..%2F..%2Fetc%2Fpasswd", "..%5C..%5Capp.py"])
def test_predict_sample_path_traversal_blocked(client, evil):
    r = client.get(f"/predict_sample/{evil}")
    assert r.status_code in (404, 400)


# ---------- POST /predict : equivalence classes of the input ----------
def test_upload_valid_pdf_ok(client, pdf_factory):
    p = pdf_factory()
    r = _post(client, open(p, "rb").read(), "valid.pdf")
    assert r.status_code == 200 and r.get_json()["verdict"] in VERDICTS


def test_upload_missing_file_field_is_400(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()


def test_upload_empty_filename_is_400(client):
    r = client.post("/predict", data={"file": (io.BytesIO(b"x"), "")}, content_type="multipart/form-data")
    assert r.status_code == 400


@pytest.mark.parametrize("name", ["note.txt", "image.png", "doc.docx", "archive.zip", "noextension"])
def test_upload_wrong_extension_rejected(client, name):
    r = _post(client, b"hello", name, "application/octet-stream")
    assert r.status_code == 400
    assert "PDF" in r.get_json()["error"]


@pytest.mark.parametrize("data", [b"", b"%PDF-1.4\n", b"this is not a pdf at all", os.urandom(2048)])
def test_upload_corrupt_or_empty_pdf_fails_gracefully(client, data):
    """Error-guessing: .pdf extension but junk content. Must be a JSON error, never a 500 stack trace page."""
    r = _post(client, data, "broken.pdf")
    assert r.status_code in (400, 500)
    assert r.is_json and "error" in r.get_json()


def test_upload_uppercase_extension_accepted(client, pdf_factory):
    p = pdf_factory()
    assert _post(client, open(p, "rb").read(), "DOC.PDF").status_code == 200


def test_upload_filename_sanitised(client, pdf_factory):
    p = pdf_factory()
    r = _post(client, open(p, "rb").read(), "../../evil name.pdf")
    assert r.status_code == 200
    assert "/" not in r.get_json()["filename"] and ".." not in r.get_json()["filename"]


# ---------- boundary values on size / pages ----------
def test_upload_multipage_pdf_uses_first_page(client, tmp_path):
    from reportlab.pdfgen import canvas
    f = tmp_path / "multi.pdf"
    c = canvas.Canvas(str(f))
    for i in range(5):
        c.drawString(60, 800, f"page {i}")
        c.showPage()
    c.save()
    assert _post(client, f.read_bytes(), "multi.pdf").status_code == 200


def test_upload_large_pdf_boundary(client, tmp_path):
    from reportlab.pdfgen import canvas
    f = tmp_path / "big.pdf"
    c = canvas.Canvas(str(f))
    for i in range(80):
        c.drawString(60, 800, "x" * 200)
        c.showPage()
    c.save()
    assert _post(client, f.read_bytes(), "big.pdf").status_code == 200


# ---------- HTTP-level ----------
@pytest.mark.parametrize("method,path", [("post", "/health"), ("post", "/samples")])
def test_wrong_http_method_405(client, method, path):
    assert getattr(client, method)(path).status_code == 405


def test_get_on_post_only_route_is_rejected(client):
    # 405 would be ideal; the app answers 404 because the static catch-all route (see D8) swallows it.
    assert client.get("/predict").status_code in (404, 405)


def test_unknown_route_404(client):
    assert client.get("/does-not-exist-xyz").status_code == 404


def test_cors_header_present(client):
    r = client.get("/health", headers={"Origin": "http://example.com"})
    assert "Access-Control-Allow-Origin" in r.headers


# ---------- repeatability / state ----------
def test_same_sample_twice_gives_same_result(client):
    a = client.get(f"/predict_sample/{SAMPLES['font']}").get_json()
    b = client.get(f"/predict_sample/{SAMPLES['font']}").get_json()
    assert a["verdict"] == b["verdict"] and a["probabilities"] == b["probabilities"]


# ---------- known defects, recorded as expected failures ----------
@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D3: UI/docs promise '100% Offline' but index.html loads Google Fonts from the network."))
def test_ui_makes_no_external_requests(client):
    html = client.get("/").data.decode()
    assert "fonts.googleapis.com" not in html


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D4: both-tampered demo sample is classified 'Metadata Tampered' (~54%), "
    "not the 'Both Tampered' its catalog entry promises."))
def test_demo_catalog_expected_verdicts_match_model(client):
    for s in client.get("/samples").get_json()["samples"]:
        got = client.get(f"/predict_sample/{s['id']}").get_json()["verdict"]
        assert got == s["expected"], s["id"]


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D5: uploaded files are saved under uploads/ and never deleted (privacy: PII retained)."))
def test_uploads_are_not_retained(client, pdf_factory):
    from app import UPLOAD_FOLDER
    p = pdf_factory()
    _post(client, open(p, "rb").read(), "retain_check.pdf")
    assert not os.path.exists(os.path.join(UPLOAD_FOLDER, "retain_check.pdf"))


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D8 (security): Flask(static_folder='.', static_url_path='') serves the WHOLE project folder: "
    "app.py, model weights, Dockerfile and uploads/ (users' PDFs + heatmaps) are downloadable by anyone. "
    "Also lets '%2e%2e%2fapp.py' via /predict_sample/ return source code."))
@pytest.mark.parametrize("path", ["/app.py", "/requirements.txt", "/src/model/predict.py",
                                  "/src/model/grogu_model.pth", "/Dockerfile",
                                  "/predict_sample/%2e%2e%2fapp.py"])
def test_server_does_not_expose_source_or_weights(client, path):
    assert client.get(path).status_code in (403, 404)


@pytest.mark.xfail(strict=True, reason="DEFECT D8: uploaded documents' heatmaps are fetchable via /uploads/...")
def test_uploaded_heatmaps_are_not_publicly_fetchable(client):
    client.get(f"/predict_sample/{SAMPLES['genuine']}")
    assert client.get("/uploads/heatmap_sample_sample_genuine_kyc.pdf.jpg").status_code in (403, 404)
