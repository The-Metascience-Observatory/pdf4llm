"""Output naming: the default must never change, and fetchpdf layouts must not collide.

These are the first tests in the repo. They deliberately cover only pure functions
and path computation -- no GROBID, no docling, no PDF parsing -- so they run in
milliseconds and can gate every commit.
"""

import inspect
from pathlib import Path

import pytest

from pdf4llm.doi_codec import doi_to_slug, slug_to_doi
from pdf4llm.naming import (
    FROM_PDF,
    already_converted,
    is_supplementary,
    output_prefix_for,
    record_stem_for,
)


# ── The regression guard ─────────────────────────────────────────────────────
# pdf4llm had no tests when output_prefix was added. Every existing caller --
# mo_pipeline, the gino_paper layout, any script globbing body.md -- depends on
# the bare names. If this test fails, that contract is broken.


def test_convert_single_still_defaults_to_bare_names():
    """No prefix argument => abstract.md / body.md / references.json, as always."""
    source = inspect.getsource(__import__("pdf4llm.pipeline", fromlist=["x"]).convert_single)
    # The prefix is empty unless explicitly supplied...
    assert 'prefix = f"{output_prefix}" if output_prefix else ""' in source
    # ...and each name is the historical one with that (empty) prefix in front.
    for name in ("abstract.md", "body.md", "references.json",
                 "references.md", "provenance.json"):
        assert f'f"{{prefix}}{name}"' in source, f"{name} lost its bare-name default"


def test_output_prefix_defaults_to_none():
    from pdf4llm.pipeline import convert_single
    params = inspect.signature(convert_single).parameters
    assert params["output_prefix"].default is None
    assert params["output_subdir_name"].default is None


def test_empty_subdir_name_is_distinct_from_none():
    """"" means 'no subfolder'; None means 'use the stem'. `or` would conflate them."""
    source = inspect.getsource(__import__("pdf4llm.pipeline", fromlist=["x"]).convert_single)
    assert "stem if output_subdir_name is None else output_subdir_name" in source
    assert "output_subdir_name or stem" not in source


# ── Prefix derivation ────────────────────────────────────────────────────────


def test_paper_prefix_is_the_encoded_doi():
    assert output_prefix_for("10.1073--pnas.2115126119.pdf") == \
        "10.1073--pnas.2115126119" + FROM_PDF


def test_paper_and_supplementary_do_not_collide():
    """The bug this whole change exists to prevent."""
    paper = output_prefix_for("10.1073--pnas.2115126119.pdf")
    si = output_prefix_for(
        "10.1073--pnas.2115126119_supplementary_info_1_pnas.2115126119.sapp.pdf")
    assert paper != si
    assert f"{paper}body.md" != f"{si}body.md"


@pytest.mark.parametrize("filename,expected_si", [
    ("10.1073--pnas.2115126119.pdf", False),
    ("10.1073--pnas.2115126119_supplementary_info_1_pnas.2115126119.sapp.pdf", True),
    # Publisher descriptors are free-form: underscores, digits, hyphens, dots.
    ("10.1038--s41562-024-02009-0_supplementary_info_2_41562_2024_2009_MOESM3_ESM.pdf", True),
    ("10.1177--0022242921990070_supplementary_info_2_10.1177_0022242921990070-img1.pdf", True),
    ("10.1093--pnasnexus--pgaf280_supplementary_info_1_pgaf280_supplementary_data.pdf", True),
])
def test_supplementary_detection(filename, expected_si):
    assert is_supplementary(filename) is expected_si


@pytest.mark.parametrize("filename", [
    "10.1073--pnas.2115126119.pdf",
    "10.1073--pnas.2115126119_supplementary_info_1_pnas.2115126119.sapp.pdf",
    "10.1073--pnas.2115126119_supplementary_info_12_anything_at_all.pdf",
])
def test_record_stem_groups_paper_with_its_supplements(filename):
    assert record_stem_for(filename) == "10.1073--pnas.2115126119"


def test_a_doi_containing_a_literal_double_hyphen_survives():
    """`10.1093/pnasnexus/pgaf280` encodes to a stem containing `--` twice."""
    stem = doi_to_slug("10.1093/pnasnexus/pgaf280")
    assert record_stem_for(f"{stem}.pdf") == stem
    assert record_stem_for(f"{stem}_supplementary_info_1_x.pdf") == stem


# ── The DOI codec must stay in sync with its two siblings ────────────────────


@pytest.mark.parametrize("doi", [
    "10.1073/pnas.2115126119",
    "10.1023/a:1015630930326",          # colon -> '~'
    "10.18260/1-2--47556",              # literal '--' inside the DOI
    "10.1002/(sici)1099-1379(199601)17:1<3::aid-job784>3.0.co;2-h",
    "10.1509/jmkr.45.6.633",
])
def test_doi_slug_round_trips(doi):
    assert slug_to_doi(doi_to_slug(doi)) == doi


def test_slug_never_contains_a_path_separator():
    assert "/" not in doi_to_slug("10.1002/(sici)1099-1379(199601)17:1<3::aid-job784>3.0.co;2-h")


def test_codec_matches_mo_pipeline_source_of_truth():
    """doi_codec.py's docstring designates mo_pipeline as the source of truth."""
    mo = pytest.importorskip("mo_pipeline.corpus.models")
    for doi in ("10.1073/pnas.2115126119", "10.1023/a:1015630930326",
                "10.18260/1-2--47556"):
        assert doi_to_slug(doi) == mo.doi_to_folder(doi)


# ── Skip-if-exists ───────────────────────────────────────────────────────────


def test_already_converted_is_false_then_true(tmp_path):
    pdf = tmp_path / "10.1073--pnas.2115126119.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert not already_converted(pdf)
    (tmp_path / f"{output_prefix_for(pdf)}body.md").write_text("# Body\n")
    assert already_converted(pdf)


def test_converting_the_paper_does_not_mark_its_supplement_done(tmp_path):
    paper = tmp_path / "10.1073--pnas.2115126119.pdf"
    si = tmp_path / "10.1073--pnas.2115126119_supplementary_info_1_x.pdf"
    for p in (paper, si):
        p.write_bytes(b"%PDF-1.4\n")
    (tmp_path / f"{output_prefix_for(paper)}body.md").write_text("# Body\n")
    assert already_converted(paper)
    assert not already_converted(si), "the SI was skipped because the paper was done"


def test_names_stay_within_the_filesystem_limit():
    """Encoded DOI + supplementary infix + a 60-char descriptor + our suffix."""
    stem = doi_to_slug("10.1002/(sici)1099-1379(199601)17:1<3::aid-job784>3.0.co;2-h")
    worst = f"{stem}_supplementary_info_9_{'d' * 60}{FROM_PDF}references.json"
    assert len(worst.encode()) < 255, len(worst.encode())
