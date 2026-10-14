"""WHITE-BOX TESTS - dataset generator (Jai).
Technique: unit tests on dataset_generator.py internals; branch coverage over the five
template branches, tamper functions, split logic. Seeds are fixed so runs are repeatable."""
import json
import os
import random
import re

import pandas as pd
import pytest

import dataset_generator as dg


@pytest.fixture(autouse=True)
def _seed():
    random.seed(7)


@pytest.fixture(autouse=True)
def _linux_fonts(monkeypatch):
    """Windows-only font paths in the repo -> fall back to any installed TTF (or Pillow default)."""
    monkeypatch.setattr(dg, "MISMATCH_FONTS", [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"])


TEMPLATES = ["T1", "T2", "T3", "T4", "T5"]


# ---------- identities ----------
@pytest.mark.parametrize("t", TEMPLATES)
def test_identity_has_core_fields(t):
    i = dg.generate_identity(t)
    for k in ("full_name", "dob", "address", "account_number", "ifsc", "pan", "base_id", "template_key"):
        assert i[k]
    assert i["template_key"] == t


def test_ifsc_account_phone_formats():
    for _ in range(40):
        i = dg.generate_identity("T1")
        assert re.fullmatch(r"[A-Z]{4}0\d{6}", i["ifsc"])
        assert re.fullmatch(r"\d{11}", i["account_number"])
        assert re.fullmatch(r"9\d{8}", i["phone"]) or re.fullmatch(r"\d{9,10}", i["phone"])
        assert re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", i["pan"])


def test_T1_transaction_balances_reconcile():
    for _ in range(25):
        i = dg.generate_identity("T1")
        bal = i["opening_balance"]
        for _d, _n, debit, credit, running in i["transactions"]:
            if debit != "-":
                bal -= int(debit.replace(",", ""))
            if credit != "-":
                bal += int(credit.replace(",", ""))
            assert bal == int(running.replace(",", ""))
        assert bal == i["closing_balance"]


def test_T1_transactions_sorted_by_date():
    i = dg.generate_identity("T1")
    dates = [pd.to_datetime(r[0], format="%d/%m/%y") for r in i["transactions"]]
    assert dates == sorted(dates)


def test_T3_units_and_amount_consistent():
    for _ in range(25):
        i = dg.generate_identity("T3")
        assert i["units"] == i["curr_reading"] - i["prev_reading"]
        assert i["amount_due"] == sum(v for _, v in i["charge_breakdown"])


def test_T5_financial_year_is_consecutive_yy_pair():
    for _ in range(40):
        fy = dg.generate_identity("T5")["financial_year"]
        a, b = fy.split("-")
        assert len(a) == 4 and len(b) == 2 and (int(a) + 1) % 100 == int(b)


def test_T4_both_subtypes_reachable():
    seen = {dg.generate_identity("T4")["doc_subtype"] for _ in range(60)}
    assert seen == {"passbook", "rental_agreement"}


# ---------- genuine rendering ----------
@pytest.mark.parametrize("t", TEMPLATES)
def test_genuine_pdf_renders_with_text_layer(t, tmp_path):
    import pymupdf
    out = tmp_path / f"{t}.pdf"
    boxes = dg.generate_genuine_pdf(dg.generate_identity(t), t, str(out))
    assert out.stat().st_size > 500
    d = pymupdf.open(str(out))
    assert len(d) == 1 and len(d[0].get_text().strip()) > 100
    assert isinstance(boxes, dict)


@pytest.mark.parametrize("t", TEMPLATES)
def test_every_template_has_tamperable_fields(t, tmp_path):
    boxes = dg.generate_genuine_pdf(dg.generate_identity(t), t, str(tmp_path / "g.pdf"))
    assert [f for f in dg.TAMPERABLE_FIELDS_BY_TEMPLATE[t] if f in boxes]


# ---------- visual tampering ----------
@pytest.mark.parametrize("sev", ["low", "medium", "high"])
@pytest.mark.parametrize("t", TEMPLATES)
def test_visual_tampering_produces_image_only_pdf(t, sev, tmp_path):
    import pymupdf
    g, o = tmp_path / "g.pdf", tmp_path / "o.pdf"
    boxes = dg.generate_genuine_pdf(dg.generate_identity(t), t, str(g))
    info = dg.apply_visual_tampering(str(g), str(o), boxes, t, sev)
    assert info["tampered_field"] in boxes and info["new_value"] != ""
    d = pymupdf.open(str(o))
    assert d[0].get_text().strip() == "" and len(d[0].get_images()) == 1


def test_fake_value_generators_cover_all_tamperable_fields():
    for fields in dg.TAMPERABLE_FIELDS_BY_TEMPLATE.values():
        for f in fields:
            assert str(dg.generate_fake_field_value(f)).strip()


# ---------- metadata tampering ----------
def test_metadata_tamper_corrupts_2_to_7_attributes(pdf_factory, tmp_path):
    for i in range(15):
        src = pdf_factory(f"s{i}.pdf")
        info = dg.apply_metadata_tampering(src, str(tmp_path / f"o{i}.pdf"), claimed_year=2019)
        assert 2 <= info["num_attributes_corrupted"] <= 7
        assert len(info["corrupted_attributes"]) == info["num_attributes_corrupted"]


def test_metadata_tamper_output_is_valid_pdf(pdf_factory, tmp_path):
    from pypdf import PdfReader
    out = tmp_path / "o.pdf"
    dg.apply_metadata_tampering(pdf_factory(), str(out), claimed_year=2018)
    assert len(PdfReader(str(out)).pages) == 1


def test_temporal_delta_matches_claimed_year(pdf_factory, tmp_path):
    """When both Producer and CreationDate get corrupted the delta = software year - claimed year."""
    hits = 0
    for i in range(60):
        info = dg.apply_metadata_tampering(pdf_factory(f"s{i}.pdf"), str(tmp_path / f"o{i}.pdf"), claimed_year=2019)
        if info["temporal_anachronism_delta"] != 0:
            hits += 1
            assert info["temporal_anachronism_delta"] > 0
    assert hits > 0


# ---------- extraction on generated data ----------
def test_genuine_reportlab_pdf_features(tmp_path):
    g = tmp_path / "g.pdf"
    dg.generate_genuine_pdf(dg.generate_identity("T2"), "T2", str(g))
    v = dg.extract_metadata_vector(str(g))
    assert v[0] == v[1] == v[2] == 1 and v[12] == 0


# ---------- end-to-end mini pipeline + split leakage ----------
def test_mini_pipeline_split_has_no_base_id_leakage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("n_genuine", "n_font_tampered", "n_metadata_tampered", "n_both_tampered"):
        monkeypatch.setitem(dg.CONFIG, k, 10)
    monkeypatch.setitem(dg.CONFIG, "output_dir", "ds")
    records, log = dg.run_pipeline()
    dg.save_outputs(records, log)
    tr, va, te = (pd.read_csv(f"ds/splits/{s}.csv") for s in ("train", "val", "test"))
    ids = [set(x.base_id) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(tr) + len(va) + len(te) == 40
    assert log["errors"] == 0
    assert json.load(open("ds/generation_log.json"))["leakage_check"] == "PASSED"
    assert set(pd.read_csv("ds/labels/master_labels.csv")["class"]) == {
        "genuine", "font_tampered", "metadata_tampered", "both_tampered"}


# ---------- known defects ----------
@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D6: dataset is NOT reproducible - random.seed(42) is set but Faker is unseeded, "
    "so names/IDs differ between runs despite the 'full reproducibility' claim."))
def test_generation_is_reproducible():
    random.seed(42)
    a = dg.generate_identity("T1")["full_name"]
    random.seed(42)
    b = dg.generate_identity("T1")["full_name"]
    assert a == b


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D7: genuine vs font-tampered classes are separable by file metadata alone "
    "(font-tampered PDFs are image-only img2pdf files with no Creator/Producer/Author)."))
def test_font_tampered_metadata_indistinguishable_from_genuine(tmp_path):
    g, g2, o = tmp_path / "g.pdf", tmp_path / "g2.pdf", tmp_path / "o.pdf"
    ident = dg.generate_identity("T1")
    boxes = dg.generate_genuine_pdf(ident, "T1", str(g))
    dg.apply_visual_tampering(str(g), str(o), boxes, "T1", "medium")
    assert dg.extract_metadata_vector(str(g))[:3] == dg.extract_metadata_vector(str(o))[:3]
