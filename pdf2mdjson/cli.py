"""
Command-line interface for PDF-to-Markdown converter.

Usage:
    pdf2md convert paper.pdf -o output/
    pdf2md batch ./pdfs/ -o ./markdown/ --workers 4
    pdf2md batch ./pdfs/ -o ./markdown/ --mode fast
    pdf2md batch ./pdfs/ -o ./markdown/ --resume
    pdf2md health-check
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import click
import requests

from .config import Config, ExtractionMode
from .pipeline import convert_single, BatchProcessor, check_grobid_health

# Default GROBID source directory (sibling of pdf_to_markdown_and_json package)
_DEFAULT_GROBID_DIR = Path(__file__).resolve().parent.parent.parent / "grobid"


def _find_grobid_dir(grobid_home: str = None) -> Path:
    """Find the GROBID source directory."""
    if grobid_home:
        p = Path(grobid_home)
        if (p / "gradlew").exists():
            return p
        raise click.ClickException(f"GROBID not found at {p} (no gradlew)")
    if (_DEFAULT_GROBID_DIR / "gradlew").exists():
        return _DEFAULT_GROBID_DIR
    raise click.ClickException(
        f"GROBID source not found at {_DEFAULT_GROBID_DIR}. "
        "Use --grobid-home to specify the GROBID source directory."
    )


def _start_grobid(grobid_dir: Path, port: int = 8070) -> tuple[subprocess.Popen, bool]:
    """Start GROBID service with DeLFT support as a background process.

    Returns:
        tuple: (process, using_delft) where using_delft is True if DeLFT models are enabled
    """

    # Check if we should use the JAR directly (for DeLFT with proper JEP path)
    jar_path = grobid_dir / "grobid-service" / "build" / "libs"
    jar_files = list(jar_path.glob("grobid-service-*-onejar.jar")) if jar_path.exists() else []

    # Look for JEP native library
    jep_paths = [
        Path.home() / ".local" / "lib" / "python3.12" / "site-packages" / "jep",
        Path.home() / ".local" / "lib" / "python3.11" / "site-packages" / "jep",
        Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "jep",
    ]
    jep_lib_path = None
    for jep_path in jep_paths:
        if jep_path.exists() and (jep_path / "libjep.so").exists():
            jep_lib_path = str(jep_path)
            break

    config_path = grobid_dir / "grobid-home" / "config" / "grobid.yaml"

    env = os.environ.copy()
    # Clear CONDA_PREFIX to avoid build.gradle JEP path detection failure
    env.pop("CONDA_PREFIX", None)
    env.pop("VIRTUAL_ENV", None)

    # If we have a JAR file and JEP library, use direct Java command for DeLFT support
    if jar_files and jep_lib_path:
        jar_file = jar_files[0]
        click.echo(f"Starting GROBID with DeLFT from {grobid_dir}...")
        click.secho("✓ Using DeLFT deep learning models (highest accuracy)", fg="green", bold=True)
        java_cmd = [
            "java",
            "-Xmx6g",
            f"-Djava.library.path={jep_lib_path}",
            "-jar", str(jar_file),
            "server", str(config_path)
        ]
        proc = subprocess.Popen(
            java_cmd,
            cwd=str(grobid_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return proc, True
    else:
        # Fallback to gradlew (CRF-only mode)
        gradlew = grobid_dir / "gradlew"
        if not gradlew.exists():
            raise click.ClickException(
                f"GROBID gradlew not found at {gradlew}. "
                "Please ensure GROBID is built first."
            )
        click.secho("\n" + "="*70, fg="red", bold=True)
        click.secho("⚠  WARNING: DeLFT NOT AVAILABLE - Using CRF-only mode", fg="red", bold=True)
        click.secho("   This is NOT the most accurate GROBID version!", fg="red", bold=True)
        click.secho("="*70 + "\n", fg="red", bold=True)

        # Show why DeLFT isn't available
        if not jar_files:
            click.secho(f"   Reason: GROBID JAR not found at {jar_path}", fg="yellow")
            click.secho("   Solution: Build GROBID with: cd grobid && ./gradlew clean install", fg="yellow")
        elif not jep_lib_path:
            click.secho("   Reason: JEP library (libjep.so) not found", fg="yellow")
            click.secho("   Solution: Install JEP with: pip install jep", fg="yellow")

        click.echo(f"\nStarting GROBID (CRF mode) from {grobid_dir}...")
        proc = subprocess.Popen(
            [str(gradlew), "run"],
            cwd=str(grobid_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return proc, False


def _wait_for_grobid(url: str, timeout: int = 120) -> bool:
    """Poll GROBID until it responds or timeout is reached."""
    deadline = time.time() + timeout
    alive_url = f"{url}/api/isalive"
    while time.time() < deadline:
        try:
            r = requests.get(alive_url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def _stop_grobid(proc: subprocess.Popen):
    """Gracefully stop the GROBID process."""
    if proc and proc.poll() is None:
        click.echo("Stopping GROBID...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """PDF-to-Markdown converter for LLM analysis.

    Convert scientific PDFs to structured Markdown optimized for
    LLM processing in metascience research.
    """
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", type=click.Path(),
              default=".", help="Output directory")
@click.option("--mode", type=click.Choice(["full-grobid", "hybrid", "fast"]),
              default="full-grobid", help="Extraction mode")
@click.option("--grobid-url", default="http://localhost:8070",
              help="GROBID server URL")
@click.option("--no-json", is_flag=True, help="Don't output JSON file")
@click.option("--save-tei", is_flag=True, help="Save raw TEI XML")
@click.option("--extract-charts", is_flag=True,
              help="Extract and analyze charts/figures using vision model (requires Ollama)")
@click.option("--chart-model", default=None,
              help="Vision model for chart analysis (default: llava:13b)")
@click.option("--ollama-url", default="http://localhost:11434",
              help="Ollama server URL for chart extraction")
@click.option("--grobid-home", type=click.Path(), default=None,
              help="Path to GROBID source directory (for auto-start)")
@click.option("--no-auto-grobid", is_flag=True,
              help="Don't auto-start GROBID (assume it's already running). By default, GROBID is auto-started with DeLFT if available.")
@click.option("--docker", "use_docker", is_flag=True,
              help="Use Docker-based GROBID (requires docker/docker-compose.yml)")
@click.option("--docker-mode", type=click.Choice(["delft", "crf"]), default="delft",
              help="Docker GROBID mode: delft (high accuracy) or crf (fastest)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def convert(pdf_path, output_dir, mode, grobid_url, no_json, save_tei,
            extract_charts, chart_model, ollama_url, grobid_home, no_auto_grobid,
            use_docker, docker_mode, verbose):
    """Convert a single PDF to Markdown.

    GROBID is auto-started with DeLFT (highest accuracy) if not already running.
    Use --no-auto-grobid to disable auto-start.

    Example:
        pdf2md convert paper.pdf -o output/
        pdf2md convert paper.pdf -o output/ --extract-charts
    """
    setup_logging(verbose)

    # Check Ollama if chart extraction requested
    if extract_charts:
        from .extractors.charts import check_ollama_available
        available, error = check_ollama_available(ollama_url)
        if not available:
            click.secho("Chart extraction requires Ollama:", fg="red")
            click.echo(error)
            sys.exit(1)
        click.echo(f"Chart extraction enabled (model: {chart_model or 'llava:13b'})")

    config = Config(
        mode=ExtractionMode.from_string(mode),
        grobid_url=grobid_url,
        output_json=not no_json,
        output_tei=save_tei,
        extract_charts=extract_charts,
        chart_model=chart_model,
        ollama_url=ollama_url,
        verbose=verbose,
    )

    # Handle Docker GROBID startup if requested
    if use_docker and config.requires_grobid():
        docker_dir = Path(__file__).parent.parent / "docker"
        docker_compose = docker_dir / "docker-compose.yml"

        if not docker_compose.exists():
            click.secho(
                f"Docker config not found at {docker_compose}\n"
                "Run from pdf2mdjson root directory or use --grobid-url to connect to existing GROBID",
                fg="red"
            )
            sys.exit(1)

        click.echo(f"Starting GROBID Docker ({docker_mode} mode)...")
        try:
            # Stop any existing containers
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose), "down"],
                capture_output=True, check=False
            )

            # Start GROBID with selected profile
            result = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose),
                 "--profile", docker_mode, "up", "-d"],
                check=True, capture_output=True, text=True
            )

            # Wait for readiness
            click.echo("Waiting for GROBID to be ready...", nl=False)
            if _wait_for_grobid(config.grobid_url, timeout=120):
                click.secho(" OK", fg="green")

                # Get version
                from .extractors.grobid import GrobidClient
                client = GrobidClient(config)
                version = client.get_version()
                if version:
                    click.echo(f"GROBID version: {version}")
                click.echo(f"Mode: {docker_mode} ({'DeLFT deep learning' if docker_mode == 'delft' else 'CRF-only (fastest)'})")
            else:
                click.secho(" TIMEOUT", fg="red")
                subprocess.run(
                    ["docker", "compose", "-f", str(docker_compose), "down"],
                    capture_output=True, check=False
                )
                sys.exit(1)

        except subprocess.CalledProcessError as e:
            click.secho(
                f"Failed to start GROBID Docker: {e.stderr if e.stderr else str(e)}",
                fg="red"
            )
            sys.exit(1)
        except FileNotFoundError:
            click.secho(
                "Docker or docker-compose not found. Please install Docker first.",
                fg="red"
            )
            sys.exit(1)

        # Skip normal GROBID startup logic
        no_auto_grobid = True

    # GROBID lifecycle management
    grobid_proc = None
    using_delft = False

    if config.requires_grobid():
        from .extractors.grobid import GrobidClient
        click.echo(f"Checking GROBID at {config.grobid_url}...", nl=False)
        client = GrobidClient(config)
        already_running = client.is_alive(timeout=5)

        if already_running:
            click.secho(" OK", fg="green")
            # Check if it's using DeLFT by trying to detect from version or config
            # For now, we'll assume pre-running GROBID might be using DeLFT
            click.echo("Note: Using existing GROBID instance (unable to verify if DeLFT is enabled)")
            using_delft = None  # Unknown
        elif no_auto_grobid:
            click.secho(" NOT AVAILABLE", fg="red")
            sys.exit(1)
        else:
            click.secho(" not running", fg="yellow")
            try:
                grobid_dir = _find_grobid_dir(grobid_home)
                grobid_proc, using_delft = _start_grobid(grobid_dir)
                click.echo(f"Waiting for GROBID to be ready...", nl=False)
                if _wait_for_grobid(config.grobid_url, timeout=120):
                    click.secho(" OK", fg="green")
                else:
                    click.secho(" TIMEOUT", fg="red")
                    _stop_grobid(grobid_proc)
                    sys.exit(1)
            except click.ClickException as e:
                click.secho(f"\n{e.message}", fg="red")
                sys.exit(1)

    click.echo(f"Converting {pdf_path} (mode: {mode})")

    try:
        result = convert_single(
            pdf_path=Path(pdf_path),
            output_dir=Path(output_dir),
            config=config,
        )
    finally:
        if grobid_proc:
            _stop_grobid(grobid_proc)
        # Note: Docker GROBID is left running for potential subsequent commands
        # Use: docker compose -f docker/docker-compose.yml down to stop

    if result.status == "success":
        click.secho(f"Success: {result.output_dir}/", fg="green")
        click.echo("  abstract.md, body.md, references.json, tables.md")
    elif result.status == "partial":
        click.secho(f"Partial success: {result.output_dir}/", fg="yellow")
        for warning in result.warnings:
            click.echo(f"  Warning: {warning}")
    else:
        click.secho("Failed", fg="red")
        for error in result.errors:
            click.echo(f"  Error ({error.stage}): {error.message}")
        sys.exit(1)

    click.echo(f"Processing time: {result.processing_time_seconds:.2f}s")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", type=click.Path(),
              required=True, help="Output directory")
@click.option("--mode", type=click.Choice(["full-grobid", "hybrid", "fast"]),
              default="full-grobid", help="Extraction mode")
@click.option("--grobid-url", default="http://localhost:8070",
              help="GROBID server URL")
@click.option("--workers", type=int, default=None,
              help="Number of parallel workers (default: min(10, CPU count))")
@click.option("--timeout", type=int, default=120,
              help="GROBID timeout in seconds (default: 120)")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip PDFs that already have .md and .json output files (default: on)")
@click.option("--resume", is_flag=True,
              help="Resume from checkpoint (skip already processed)")
@click.option("--checkpoint", type=click.Path(),
              help="Checkpoint file path")
@click.option("--no-json", is_flag=True, help="Don't output JSON files")
@click.option("--grobid-home", type=click.Path(), default=None,
              help="Path to GROBID source directory (for auto-start)")
@click.option("--no-auto-grobid", is_flag=True,
              help="Don't auto-start GROBID (assume it's already running). By default, GROBID is auto-started with DeLFT if available.")
@click.option("--docker", "use_docker", is_flag=True,
              help="Use Docker-based GROBID (requires docker/docker-compose.yml)")
@click.option("--docker-mode", type=click.Choice(["delft", "crf"]), default="delft",
              help="Docker GROBID mode: delft (high accuracy) or crf (fastest)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def batch(input_dir, output_dir, mode, grobid_url, workers, timeout, skip_existing,
          resume, checkpoint, no_json, grobid_home, no_auto_grobid, use_docker, docker_mode, verbose):
    """Convert a folder of PDFs to Markdown.

    GROBID is auto-started with DeLFT (highest accuracy) if not already running.
    Use --no-auto-grobid to disable auto-start.

    Example:
        pdf2md batch ./pdfs/ -o ./markdown/ --workers 4
        pdf2md batch ./pdfs/ -o ./markdown/ --resume
        pdf2md batch ./pdfs/ -o ./markdown/ --skip-existing --workers 1 --timeout 300
    """
    setup_logging(verbose)

    config_kwargs = dict(
        mode=ExtractionMode.from_string(mode),
        grobid_url=grobid_url,
        grobid_timeout=timeout,
        output_json=not no_json,
        verbose=verbose,
    )
    if workers is not None:
        config_kwargs["workers"] = workers
    config = Config(**config_kwargs)

    # Set checkpoint file
    checkpoint_file = None
    if checkpoint:
        checkpoint_file = Path(checkpoint)
    elif resume:
        checkpoint_file = Path(output_dir) / ".pdf2md_checkpoint.json"

    click.echo(f"Batch conversion (mode: {mode}, workers: {config.workers}, timeout: {timeout}s)")
    click.echo(f"Input: {input_dir}")
    click.echo(f"Output: {output_dir}")

    if skip_existing:
        click.echo("Skip existing: enabled (will skip PDFs with existing .md/.json)")
    if checkpoint_file:
        click.echo(f"Checkpoint: {checkpoint_file}")

    # Handle Docker GROBID startup if requested
    if use_docker and config.requires_grobid():
        docker_dir = Path(__file__).parent.parent / "docker"
        docker_compose = docker_dir / "docker-compose.yml"

        if not docker_compose.exists():
            click.secho(
                f"Docker config not found at {docker_compose}\n"
                "Run from pdf2mdjson root directory or use --grobid-url to connect to existing GROBID",
                fg="red"
            )
            sys.exit(1)

        click.echo(f"Starting GROBID Docker ({docker_mode} mode)...")
        try:
            # Stop any existing containers
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose), "down"],
                capture_output=True, check=False
            )

            # Start GROBID with selected profile
            result = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose),
                 "--profile", docker_mode, "up", "-d"],
                check=True, capture_output=True, text=True
            )

            # Wait for readiness
            click.echo("Waiting for GROBID to be ready...", nl=False)
            if _wait_for_grobid(config.grobid_url, timeout=120):
                click.secho(" OK", fg="green")

                # Get version
                from .extractors.grobid import GrobidClient
                client = GrobidClient(config)
                version = client.get_version()
                if version:
                    click.echo(f"GROBID version: {version}")
                click.echo(f"Mode: {docker_mode} ({'DeLFT deep learning' if docker_mode == 'delft' else 'CRF-only (fastest)'})")
            else:
                click.secho(" TIMEOUT", fg="red")
                subprocess.run(
                    ["docker", "compose", "-f", str(docker_compose), "down"],
                    capture_output=True, check=False
                )
                sys.exit(1)

        except subprocess.CalledProcessError as e:
            click.secho(
                f"Failed to start GROBID Docker: {e.stderr if e.stderr else str(e)}",
                fg="red"
            )
            sys.exit(1)
        except FileNotFoundError:
            click.secho(
                "Docker or docker-compose not found. Please install Docker first.",
                fg="red"
            )
            sys.exit(1)

        # Skip normal GROBID startup logic
        no_auto_grobid = True

    # GROBID lifecycle management
    grobid_proc = None
    using_delft = False

    if config.requires_grobid():
        from .extractors.grobid import GrobidClient
        click.echo(f"Checking GROBID at {config.grobid_url}...", nl=False)
        client = GrobidClient(config)
        already_running = client.is_alive(timeout=5)

        if already_running:
            click.secho(" OK", fg="green")
            click.echo("Note: Using existing GROBID instance (unable to verify if DeLFT is enabled)")
            using_delft = None  # Unknown
        elif no_auto_grobid:
            click.secho(" NOT AVAILABLE", fg="red")
            click.echo("GROBID is not running and --no-auto-grobid was specified.")
            sys.exit(1)
        else:
            click.secho(" not running", fg="yellow")
            try:
                grobid_dir = _find_grobid_dir(grobid_home)
                grobid_proc, using_delft = _start_grobid(grobid_dir)
                click.echo(f"Waiting for GROBID to be ready at {config.grobid_url}...", nl=False)
                if _wait_for_grobid(config.grobid_url, timeout=120):
                    click.secho(" OK", fg="green")
                else:
                    click.secho(" TIMEOUT", fg="red")
                    _stop_grobid(grobid_proc)
                    sys.exit(1)
            except click.ClickException as e:
                click.secho(f"\n{e.message}", fg="red")
                sys.exit(1)

    processor = BatchProcessor(config, checkpoint_file)

    try:
        result = processor.process_folder(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            resume=resume,
            skip_existing=skip_existing,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        if grobid_proc:
            _stop_grobid(grobid_proc)
        sys.exit(1)
    finally:
        if grobid_proc:
            _stop_grobid(grobid_proc)
        # Note: Docker GROBID is left running for potential subsequent commands
        # Use: docker compose -f docker/docker-compose.yml down to stop

    # Print summary
    click.echo("")
    click.echo(result.generate_report())

    if result.failed > 0:
        sys.exit(1)


@cli.command("health-check")
@click.option("--grobid-url", default="http://localhost:8070",
              help="GROBID server URL")
def health_check(grobid_url):
    """Check GROBID server health.

    Example:
        pdf2md health-check
        pdf2md health-check --grobid-url http://grobid:8070
    """
    config = Config(grobid_url=grobid_url)

    click.echo(f"Checking GROBID at {grobid_url}...")

    health = check_grobid_health(config)

    if health["alive"]:
        click.secho("GROBID is running", fg="green")
        click.echo(f"Version: {health['version']}")
    else:
        click.secho("GROBID is not available", fg="red")
        click.echo("")
        click.echo("To start GROBID with Docker:")
        click.echo("  docker run --gpus all -p 8070:8070 lfoppiano/grobid:0.8.0")
        click.echo("")
        click.echo("Without GPU:")
        click.echo("  docker run -p 8070:8070 lfoppiano/grobid:0.8.0")
        sys.exit(1)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--grobid-url", default="http://localhost:8070",
              help="GROBID server URL")
def test_extraction(pdf_path, grobid_url):
    """Test extraction on a single PDF and show summary.

    Example:
        pdf2md test-extraction paper.pdf
    """
    setup_logging(verbose=True)

    from .extractors.grobid import extract_with_grobid, GrobidClient

    config = Config(grobid_url=grobid_url)

    # Check GROBID
    client = GrobidClient(config)
    if not client.is_alive():
        click.secho("GROBID not available", fg="red")
        sys.exit(1)

    click.echo(f"Testing extraction on {pdf_path}")
    click.echo("")

    try:
        tei_xml, document = extract_with_grobid(Path(pdf_path), config)

        click.secho("Extraction successful!", fg="green")
        click.echo("")
        click.echo(f"Title: {document.title}")
        click.echo(f"DOI: {document.doi or '[NOT FOUND]'}")
        click.echo("")
        click.echo(f"Sections: {len(document.sections)}")
        click.echo(f"Tables: {len(document.tables)}")
        click.echo(f"Figures: {len(document.figures)}")
        click.echo(f"References: {len(document.references)}")
        click.echo("")

        # Reference DOI stats
        refs_with_doi = sum(1 for r in document.references if r.doi)
        click.echo(f"References with DOI: {refs_with_doi}/{len(document.references)}")

        if document.references:
            click.echo("")
            click.echo("Sample references:")
            for ref in document.references[:3]:
                doi_str = f"DOI:{ref.doi}" if ref.doi else "[NO-DOI]"
                click.echo(f"  {ref.num}. {ref.title or 'Unknown'}... {doi_str}")

    except Exception as e:
        click.secho(f"Extraction failed: {e}", fg="red")
        sys.exit(1)


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
