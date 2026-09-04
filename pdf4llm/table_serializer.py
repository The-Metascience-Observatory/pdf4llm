"""Docling markdown export that keeps tables as embedded HTML.

Docling's default `export_to_markdown()` renders every table as a markdown pipe
table, which **cannot express `rowspan` or `colspan`**. Clinical results tables
depend on exactly that structure — arm x timepoint grids, grouped headers
spanning several columns — and a pipe table silently flattens a merged header
cell into the wrong column. That is how an outcome gets attributed to the wrong
arm, and it is invisible downstream: the text is there, it is just wrong.

Embedding the original HTML keeps the spans intact AND is *cheaper*: measured on
a real trial paper, the HTML-table render was 116,330 chars against 121,802 for
pipe tables, because pipe tables pad every cell to a common column width.

This is the canonical implementation. `birds_eye_review_code/html_convert.py`
imports it (with a local fallback) so the two cannot drift.
"""
from __future__ import annotations

from docling_core.transforms.serializer.base import (BaseTableSerializer,
                                                     SerializationResult)


class HtmlTableSerializer(BaseTableSerializer):
    """Serialize each table as raw HTML instead of a markdown pipe table.

    Subclasses the ABSTRACT `BaseTableSerializer`, not the concrete
    `MarkdownTableSerializer`: this replaces pipe-table rendering wholesale, and
    inheriting the concrete class would silently pick its formatting back up if
    docling changed its base implementation. The base class is also required —
    `MarkdownDocSerializer` is a pydantic model that validates the type.
    """

    def serialize(self, *, item, doc_serializer=None, doc=None, **kwargs):
        try:
            html = item.export_to_html(doc=doc)
        except TypeError:          # older docling: no doc kwarg
            html = item.export_to_html()
        return SerializationResult(text="\n" + (html or "") + "\n")


def export_markdown_html_tables(doc) -> str:
    """`doc.export_to_markdown()` equivalent, with tables as embedded HTML.

    Falls back to the plain export if the serializer path raises, so a docling
    version change degrades to today's behaviour (flattened tables) rather than
    failing the conversion outright — a worse table beats no paper.
    """
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer, MarkdownParams)
    try:
        serializer = MarkdownDocSerializer(
            doc=doc,
            table_serializer=HtmlTableSerializer(),
            # escape_html=False, or the markup we just inserted comes back
            # entity-escaped as visible &lt;table&gt;.
            params=MarkdownParams(escape_html=False),
        )
        return serializer.serialize().text or ""
    except Exception:
        return doc.export_to_markdown() or ""
