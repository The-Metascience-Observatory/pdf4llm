"""Reversible DOI <-> filename-slug codec.

Vendored from mo_pipeline.corpus.models (doi_to_folder / folder_to_doi), the
source of truth — keep the two in sync. The old slug here collapsed every
non-word character to '--', which collided with the '/' encoding and made
slugs impossible to decode ('10.1023/a:123' and '10.1023/a/123' produced the
same name).

Mapping:
  '/' -> '--'          DOI path separator
  ':' -> '~'           colon (old Springer/Kluwer DOIs)
  < > " \\ | ? * -> '~XX~'  hex-escaped (Wiley SICI DOIs)
  '-' -> '~2d~'        only in a '--' run or adjacent to '/', where a literal
                       hyphen would be ambiguous with the slash encoding
                       (ASEE DOIs like 10.18260/1-2--47556)
"""
import re

_FS_FORBIDDEN = '<>"\\|?*'


def doi_to_slug(doi: str) -> str:
    s = doi.strip()
    s = re.sub(r"-+(?=/)|(?<=/)-+|-{2,}", lambda m: "~2d~" * len(m.group()), s)
    s = s.replace("/", "--")
    for ch in _FS_FORBIDDEN:
        s = s.replace(ch, f"~{ord(ch):02x}~")
    return s.replace(":", "~")


def slug_to_doi(slug: str) -> str:
    # Decode order matters: '~2d~' to a sentinel first so the '--' -> '/' pass
    # cannot touch escaped hyphens; restore them last.
    s = slug.replace("~2d~", "\x00")
    for ch in _FS_FORBIDDEN:
        s = s.replace(f"~{ord(ch):02x}~", ch)
    return s.replace("~", ":").replace("--", "/").replace("\x00", "-")
