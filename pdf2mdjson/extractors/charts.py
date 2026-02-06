"""
Chart and figure extraction module.

Extracts images from PDFs and uses vision-language models to generate
textual descriptions of charts, graphs, and figures.

Supports multiple backends (FREE options first):
1. Ollama (LOCAL - FREE, RECOMMENDED) - LLaVA, BakLLaVA, Llama-3.2-Vision
   - Runs on your GPU, completely free and private
   - Install: https://ollama.ai then `ollama pull llava:13b`
   - With 12GB GPU, use llava:13b or llama3.2-vision:11b for best results

2. Anthropic Claude - Best quality but costs money
3. OpenAI GPT-4V - Good but costs money
"""

import base64
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFigure:
    """Represents an extracted figure with its analysis."""
    figure_id: str
    page_number: int
    caption: Optional[str]
    image_data: bytes  # PNG image data
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    description: Optional[str] = None  # AI-generated description
    chart_data: Optional[str] = None  # Extracted data in markdown table format


class ChartExtractor:
    """
    Extracts and analyzes charts/figures from PDFs.

    Uses vision-language models to generate descriptions and extract data.
    """

    # Prompt for chart analysis
    CHART_ANALYSIS_PROMPT = """Analyze this figure from a scientific paper. Provide:

1. **Figure Type**: What kind of visualization is this? (bar chart, line graph, scatter plot, flow diagram, etc.)

2. **Description**: A detailed description of what the figure shows, including:
   - What variables are being compared
   - Key patterns or trends
   - Notable data points or outliers

3. **Data Extraction**: If this is a quantitative chart (bar chart, line graph, etc.), extract the data as a markdown table. Include:
   - All axis labels and values
   - All data points you can read from the chart
   - Any error bars or confidence intervals if visible

4. **Statistical Information**: Note any p-values, confidence intervals, or statistical annotations shown.

Format your response as:

**Type**: [figure type]

**Description**: [detailed description]

**Extracted Data**:
| [Column1] | [Column2] | ... |
|-----------|-----------|-----|
| [value]   | [value]   | ... |

**Statistics**: [any statistical annotations]

If you cannot extract numerical data (e.g., for diagrams or photos), explain what the figure depicts instead."""

    def __init__(
        self,
        provider: str = "ollama",  # Default to FREE local option
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize chart extractor.

        Args:
            provider: "ollama" (free, local), "anthropic", or "openai"
            api_key: API key for cloud providers (can also use env vars)
            model: Model name (defaults based on provider)
            ollama_url: URL for Ollama server if using local models

        Recommended free setup:
            1. Install Ollama: https://ollama.ai
            2. Pull a vision model: ollama pull llava:13b
            3. Use default settings (provider="ollama")
        """
        self.provider = provider.lower()
        self.ollama_url = ollama_url

        # Set default models - prioritize best free options
        if model:
            self.model = model
        elif self.provider == "ollama":
            # llava:13b is good balance of quality and speed for 12GB GPU
            # Alternatives: llama3.2-vision:11b, bakllava, llava:34b (needs more VRAM)
            self.model = "llava:13b"
        elif self.provider == "anthropic":
            self.model = "claude-sonnet-4-20250514"
        elif self.provider == "openai":
            self.model = "gpt-4o"
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            self.api_key = None

        # Validate configuration
        if self.provider in ("anthropic", "openai") and not self.api_key:
            raise ValueError(
                f"API key required for {provider}. "
                f"Set {provider.upper()}_API_KEY environment variable or pass api_key parameter."
            )

    def extract_figures_from_pdf(
        self,
        pdf_path: Path,
        min_width: int = 100,
        min_height: int = 100,
    ) -> List[ExtractedFigure]:
        """
        Extract figure images from a PDF.

        Args:
            pdf_path: Path to PDF file
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract

        Returns:
            List of ExtractedFigure objects with image data
        """
        figures = []

        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            # Get images on this page
            image_list = page.get_images()

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Skip small images (likely icons or decorations)
                    if width < min_width or height < min_height:
                        continue

                    # Convert to PNG if needed
                    if image_ext != "png":
                        image_bytes = self._convert_to_png(image_bytes, image_ext)

                    # Get image position on page
                    bbox = self._get_image_bbox(page, xref)

                    figure_id = f"fig_{page_num + 1}_{img_idx + 1}"

                    figures.append(ExtractedFigure(
                        figure_id=figure_id,
                        page_number=page_num + 1,
                        caption=None,  # Will be filled from GROBID data if available
                        image_data=image_bytes,
                        bbox=bbox,
                    ))

                except Exception as e:
                    logger.warning(f"Failed to extract image {xref} from page {page_num}: {e}")

        doc.close()

        logger.info(f"Extracted {len(figures)} figures from {pdf_path}")
        return figures

    def _convert_to_png(self, image_bytes: bytes, ext: str) -> bytes:
        """Convert image to PNG format."""
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    def _get_image_bbox(self, page, xref: int) -> Tuple[float, float, float, float]:
        """Get bounding box for an image on the page."""
        # Try to find image rectangle
        for img in page.get_images():
            if img[0] == xref:
                # Get image rectangles
                rects = page.get_image_rects(img)
                if rects:
                    rect = rects[0]
                    return (rect.x0, rect.y0, rect.x1, rect.y1)
        return (0, 0, 0, 0)

    def analyze_figure(self, figure: ExtractedFigure) -> ExtractedFigure:
        """
        Analyze a figure using vision-language model.

        Args:
            figure: ExtractedFigure with image data

        Returns:
            Updated ExtractedFigure with description and chart_data
        """
        if self.provider == "anthropic":
            description = self._analyze_with_anthropic(figure.image_data)
        elif self.provider == "openai":
            description = self._analyze_with_openai(figure.image_data)
        elif self.provider == "ollama":
            description = self._analyze_with_ollama(figure.image_data)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        figure.description = description

        # Try to extract just the data table if present
        figure.chart_data = self._extract_data_table(description)

        return figure

    def _analyze_with_anthropic(self, image_data: bytes) -> str:
        """Analyze image using Anthropic Claude."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        # Encode image as base64
        image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

        message = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.CHART_ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )

        return message.content[0].text

    def _analyze_with_openai(self, image_data: bytes) -> str:
        """Analyze image using OpenAI GPT-4V."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        # Encode image as base64
        image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": self.CHART_ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content

    def _analyze_with_ollama(self, image_data: bytes) -> str:
        """Analyze image using Ollama (local LLaVA model)."""
        import requests

        # Encode image as base64
        image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": self.CHART_ANALYSIS_PROMPT,
                "images": [image_b64],
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama request failed: {response.text}")

        return response.json()["response"]

    def _extract_data_table(self, description: str) -> Optional[str]:
        """Extract markdown table from description if present."""
        lines = description.split("\n")
        in_table = False
        table_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                in_table = True
                table_lines.append(stripped)
            elif in_table and not stripped.startswith("|"):
                # End of table
                break

        if table_lines:
            return "\n".join(table_lines)
        return None

    def analyze_all_figures(
        self,
        figures: List[ExtractedFigure],
        progress_callback=None,
    ) -> List[ExtractedFigure]:
        """
        Analyze all figures in a list.

        Args:
            figures: List of ExtractedFigure objects
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of figures with descriptions added
        """
        analyzed = []
        total = len(figures)

        for idx, figure in enumerate(figures):
            try:
                logger.info(f"Analyzing figure {figure.figure_id} ({idx + 1}/{total})")
                analyzed_figure = self.analyze_figure(figure)
                analyzed.append(analyzed_figure)

                if progress_callback:
                    progress_callback(idx + 1, total)

            except Exception as e:
                logger.error(f"Failed to analyze figure {figure.figure_id}: {e}")
                figure.description = f"[Analysis failed: {e}]"
                analyzed.append(figure)

        return analyzed


def check_ollama_available(ollama_url: str = "http://localhost:11434") -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama is running and has a vision model available.

    Returns:
        Tuple of (is_available, error_message)
    """
    import requests

    try:
        # Check if Ollama is running
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server not responding"

        # Check for vision models
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        vision_models = [
            m for m in model_names
            if any(v in m.lower() for v in ["llava", "vision", "bakllava"])
        ]

        if not vision_models:
            return False, (
                "No vision models found. Install one with:\n"
                "  ollama pull llava:13b\n"
                "Or for your 12GB GPU:\n"
                "  ollama pull llama3.2-vision:11b"
            )

        return True, None

    except requests.exceptions.ConnectionError:
        return False, (
            "Ollama not running. Install and start it:\n"
            "  1. Install from https://ollama.ai\n"
            "  2. Run: ollama serve\n"
            "  3. Pull a model: ollama pull llava:13b"
        )
    except Exception as e:
        return False, f"Error checking Ollama: {e}"


def extract_and_analyze_charts(
    pdf_path: Path,
    provider: str = "ollama",  # Default to FREE local option
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
) -> List[ExtractedFigure]:
    """
    Convenience function to extract and analyze all charts from a PDF.

    Args:
        pdf_path: Path to PDF file
        provider: Vision model provider - "ollama" (free), "anthropic", or "openai"
        api_key: API key for cloud providers (not needed for ollama)
        model: Model name (optional, uses defaults)
        ollama_url: URL for Ollama server

    Returns:
        List of ExtractedFigure objects with descriptions

    Example (free, local):
        # First: ollama pull llava:13b
        figures = extract_and_analyze_charts("paper.pdf")
    """
    extractor = ChartExtractor(
        provider=provider,
        api_key=api_key,
        model=model,
        ollama_url=ollama_url,
    )
    figures = extractor.extract_figures_from_pdf(pdf_path)

    if figures:
        figures = extractor.analyze_all_figures(figures)

    return figures
