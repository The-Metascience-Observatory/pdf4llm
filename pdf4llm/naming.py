"""Output-name derivation for PDFs laid out by fetchpdf.

fetchpdf writes one directory per record, every artifact sharing the encoded-DOI
stem::

    10.1073--pnas.2115126119/
        10.1073--pnas.2115126119.pdf
        10.1073--pnas.2115126119.xml
        10.1073--pnas.2115126119_from_xml.md
        10.1073--pnas.2115126119_supplementary_info_1_pnas.2115126119.sapp.pdf

Converting the paper and its supplementary PDFs into that same directory needs a
per-file prefix, or the second conversion overwrites the first's ``body.md``.

Two rules, both inherited rather than invented:

1. **Provenance lives in the filename.** fetchpdf already writes ``_from_xml.md``
   and ``_from_html.md``, deliberately, because a consumer that globs a directory
   reads nothing else -- and publisher JATS and a reconstructed PDF do not deserve
   equal trust. ``_from_pdf`` is the free slot in that scheme.

2. **Derive by suffix replacement, never by re-parsing the DOI.** The supplementary
   descriptor is free-form publisher text that can contain dots, hyphens and
   underscores, so a regex that tries to pull ``_SI_1`` back out of it will
   eventually mangle a name. Replacing the trailing ``.pdf`` keeps the
   ``_supplementary_info_{N}`` anchor intact, which is what the existing manifest
   keys on.
"""

import re
from pathlib import Path

#: Marks markdown reconstructed from a PDF, as opposed to fetchpdf's `_from_xml.md`
#: / `_from_html.md`. PDF is the lowest-fidelity source in the tier ladder: its
#: structure is reconstructed from glyph positions, so numeric corruption is
#: possible and undetectable. The reader must be able to see that at a glance.
FROM_PDF = "_from_pdf_"

#: fetchpdf's supplementary infix, `{stem}_supplementary_info_{N}{descriptor}{ext}`.
SUPPL_RE = re.compile(r"^(?P<stem>.+?)_supplementary_info_(?P<index>\d+)(?:[._].*)?$")


def output_prefix_for(pdf_path) -> str:
    """The prefix for every artifact converted from `pdf_path`.

    ``10.1073--x.pdf``                          -> ``10.1073--x_from_pdf_``
    ``10.1073--x_supplementary_info_1_y.pdf``   -> ``10.1073--x_supplementary_info_1_y_from_pdf_``

    The whole stem is kept for supplementary files rather than reduced to an
    index, so the name still says which publisher file it came from.
    """
    return f"{Path(pdf_path).stem}{FROM_PDF}"


def is_supplementary(pdf_path) -> bool:
    """True when this PDF is a supplementary artifact rather than the paper."""
    return SUPPL_RE.match(Path(pdf_path).stem) is not None


def record_stem_for(pdf_path) -> str:
    """The encoded-DOI stem of the record a PDF belongs to.

    Both the paper and its supplementary files map to the same record stem, which
    is what lets a caller group them::

        10.1073--x.pdf                        -> 10.1073--x
        10.1073--x_supplementary_info_1_y.pdf -> 10.1073--x
    """
    stem = Path(pdf_path).stem
    match = SUPPL_RE.match(stem)
    return match.group("stem") if match else stem


def already_converted(pdf_path) -> bool:
    """True when the body markdown for this PDF is already on disk beside it."""
    pdf_path = Path(pdf_path)
    return (pdf_path.parent / f"{output_prefix_for(pdf_path)}body.md").exists()
