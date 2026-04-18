import pandas as pd
import html

def format_authors(author_str):
    """Return 'Last F. et al.' style from semicolon-separated author list."""
    if not isinstance(author_str, str) or not author_str.strip():
        return ""
    first_author = author_str.split(";")[0].strip()
    parts = first_author.split()
    if len(parts) >= 2:
        last = parts[-1]
        first = parts[0][0] + "."
        name = f"{last} {first}"
        return f"{name} <i>et al.</i>"
    else:
        name = parts[0]
        return f"{name}"

def generate_citation_html_for_website(authors, journal, year, doi):
    """Return clickable HTML citation (whole citation linked, DOI hidden)."""
    authors_part = format_authors(authors)
    journal_part = html.escape(journal) if isinstance(journal, str) else ""
    year_part = str(int(year)) if pd.notna(year) else ""

    # Combine citation text
    citation_text = " ".join(p for p in [authors_part, f"<i>{journal_part}</i>", year_part] if p)

    # Wrap the entire citation in a hyperlink if DOI exists
    if isinstance(doi, str) and doi.strip():
        url = f"https://doi.org/{doi.strip()}"
        citation_html = f'<a href="{html.escape(url)}" target="_blank" style="text-decoration:none; color:inherit;">{citation_text}</a>'
    else:
        citation_html = citation_text

    return citation_html
