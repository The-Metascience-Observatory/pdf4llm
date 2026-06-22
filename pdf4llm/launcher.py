"""Thin stdlib-only launcher for the pdf4llm CLI.

This module is registered as the ``pdf4llm`` console script. Its job is to
make sure pdf4llm runs inside an isolated venv (so the user's base Python /
anaconda environment can't break it via dependency conflicts), then re-exec
into that venv's interpreter to run the real CLI at ``pdf4llm.cli:main``.

It deliberately imports ONLY stdlib modules — no click, no pydantic, no
pdf4llm submodules — so it can run cleanly in any Python environment, even
one with a broken numpy ABI like the one this codebase has been hitting.

Behavior:
  • If ``PDF4LLM_VENV_ACTIVE=1`` or ``--no-venv``  → run the CLI in-process.
  • If ``VIRTUAL_ENV`` is set (user activated a venv themselves) → run in-process.
  • Else use the managed venv at the path stored in ``~/.pdf4llm/config.json``;
    if no config or no usable venv, prompt the user for a path and create it.
  • Then ``os.execvpe`` into ``<venv>/bin/python -m pdf4llm.cli ...``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ----- constants -----

CONFIG_PATH = Path.home() / ".pdf4llm" / "config.json"
READY_MARKER = ".pdf4llm-ready"     # written inside the venv after a successful install
DEFAULT_VENV_GB = 4.0                # rough disk estimate for torch+docling+transformers+pandas
MIN_FREE_GB = 5.0                    # refuse to create the venv if free disk < this


# ----- repo / default-venv discovery -----

def _repo_root() -> Path:
    """Path to the pdf4llm source tree (sibling-of-package directory)."""
    return Path(__file__).resolve().parent.parent


def _default_venv_path() -> Path:
    return _repo_root() / ".venv"


# ----- config file -----

def _read_config() -> dict:
    try:
        with CONFIG_PATH.open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(cfg, f, indent=2)
    tmp.replace(CONFIG_PATH)


# ----- venv lifecycle -----

def _venv_python(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if os.name == "nt" else "bin") / "python"


def _venv_is_usable(venv_path: Path) -> bool:
    """Venv must exist, have a python interpreter, and be marked ready."""
    return (
        venv_path.is_dir()
        and _venv_python(venv_path).exists()
        and (venv_path / READY_MARKER).exists()
    )


def _disk_free_gb(path: Path) -> float:
    target = path
    while target and not target.exists():
        target = target.parent
    try:
        return shutil.disk_usage(target).free / (1024 ** 3)
    except OSError:
        return float("inf")


def _prompt_venv_path() -> Path:
    """Interactive first-run prompt. Returns absolute Path; never returns None."""
    default = _default_venv_path()
    free = _disk_free_gb(default.parent)
    print()
    print(f"pdf4llm needs an isolated Python environment (~{DEFAULT_VENV_GB:.0f} GB on disk).")
    print(f"This keeps its heavy ML deps (torch, docling, transformers) from")
    print(f"colliding with packages in your base Python environment.")
    print(f"Disk free: {free:.1f} GB")
    if free < MIN_FREE_GB:
        print(
            f"ERROR: only {free:.1f} GB free — need at least {MIN_FREE_GB:.0f} GB.",
            file=sys.stderr,
        )
        sys.exit(1)
    while True:
        raw = input(f"Venv location [{default}]: ").strip()
        chosen = Path(raw).expanduser().resolve() if raw else default.resolve()
        if chosen.exists() and not chosen.is_dir():
            print(f"  {chosen} exists and is not a directory; pick another.")
            continue
        return chosen


def _create_venv(venv_path: Path) -> None:
    print(f"Creating venv at {venv_path}...")
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the python that's running this launcher. `--upgrade-deps` ensures
    # pip/setuptools in the venv are fresh enough to install modern wheels.
    subprocess.run(
        [sys.executable, "-m", "venv", "--upgrade-deps", str(venv_path)],
        check=True,
    )


def _install_pdf4llm_into_venv(venv_path: Path) -> None:
    """Install pdf4llm (editable) and all its deps into the venv."""
    python = _venv_python(venv_path)
    repo = _repo_root()
    print(f"Installing pdf4llm + dependencies into {venv_path}")
    print("(first install pulls torch, docling, transformers — takes a few minutes)")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-e", str(repo)],
        check=True,
    )
    (venv_path / READY_MARKER).write_text("ok\n")
    print(f"✓ pdf4llm ready in {venv_path}")


# ----- launcher arg parsing -----

def _split_launcher_args(argv: list) -> tuple[argparse.Namespace, list]:
    """Pull launcher-only flags out of argv; forward the rest to the CLI.

    Launcher flags:
      --venv PATH          override the venv location for this run (and save).
      --no-venv            don't use a venv; run the CLI in this interpreter.
      --reinstall-venv     re-run `pip install -e .` inside the venv.
      --venv-status        print the managed-venv path and exit.

    Any flag this parser doesn't know is forwarded verbatim — including the
    common case of ``pdf4llm somefile.pdf``.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--venv", type=str, default=None)
    parser.add_argument("--no-venv", action="store_true")
    parser.add_argument("--reinstall-venv", action="store_true")
    parser.add_argument("--venv-status", action="store_true")
    args, rest = parser.parse_known_args(argv)
    return args, rest


def _invoke_cli_in_process(argv: list) -> None:
    """Run pdf4llm.cli:main() inside this interpreter."""
    # Imported lazily so we never touch the heavy deps in the wrong interpreter.
    from pdf4llm.cli import main as cli_main
    sys.argv = ["pdf4llm", *argv]
    cli_main()


def main() -> None:
    args, forwarded = _split_launcher_args(sys.argv[1:])

    if args.venv_status:
        cfg = _read_config()
        venv = cfg.get("venv_path") or "(not configured)"
        in_venv = os.environ.get("VIRTUAL_ENV") or "(none)"
        print(f"Managed venv:      {venv}")
        print(f"Currently in venv: {in_venv}")
        print(f"Re-exec guard set: {os.environ.get('PDF4LLM_VENV_ACTIVE') == '1'}")
        return

    # 1. Already in the right interpreter — just run the CLI.
    if os.environ.get("PDF4LLM_VENV_ACTIVE") == "1" or args.no_venv:
        _invoke_cli_in_process(forwarded)
        return

    # 2. User activated some venv themselves — respect it.
    if os.environ.get("VIRTUAL_ENV"):
        _invoke_cli_in_process(forwarded)
        return

    # 3. Resolve / create the managed venv.
    cfg = _read_config()
    if args.venv:
        venv_path = Path(args.venv).expanduser().resolve()
    elif cfg.get("venv_path"):
        venv_path = Path(cfg["venv_path"]).expanduser().resolve()
    else:
        venv_path = _prompt_venv_path()

    if not _venv_is_usable(venv_path):
        if not _venv_python(venv_path).exists():
            _create_venv(venv_path)
        _install_pdf4llm_into_venv(venv_path)
    elif args.reinstall_venv:
        _install_pdf4llm_into_venv(venv_path)

    _write_config({**cfg, "venv_path": str(venv_path)})

    # 4. Re-exec into the venv's python. PDF4LLM_VENV_ACTIVE=1 prevents an
    #    infinite loop if something inside the venv re-enters this launcher.
    python = _venv_python(venv_path)
    if not python.exists():
        print(f"ERROR: venv python missing at {python}", file=sys.stderr)
        sys.exit(1)
    env = os.environ.copy()
    env["PDF4LLM_VENV_ACTIVE"] = "1"
    os.execvpe(str(python), [str(python), "-m", "pdf4llm.cli", *forwarded], env)


if __name__ == "__main__":
    main()
