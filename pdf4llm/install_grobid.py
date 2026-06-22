"""Install and start GROBID for use by pdf4llm.

Two install paths are supported:

* **source** — clone the GROBID repo to ``~/.pdf4llm/grobid`` and build it
  with the bundled Gradle wrapper. Requires Java 11+ and git. ~3 GB on disk
  (most of it Gradle deps + GROBID home models). First build can take
  10–20 minutes. Runs natively — no Docker.

* **docker** — pull the prebuilt GROBID image (delft ~12 GB or crf ~500 MB)
  and run it via the bundled ``docker/docker-compose.yml``.

Exposed as the ``pdf4llm-install-grobid`` console script, and reused
internally by the CLI to offer one-time installation when GROBID is missing.
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click


def _brew_env() -> dict:
    """Environment for non-interactive Homebrew (skips auto-update + confirm prompts)."""
    env = os.environ.copy()
    env["NONINTERACTIVE"] = "1"
    env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
    env["HOMEBREW_NO_INSTALL_CLEANUP"] = "1"
    env["HOMEBREW_NO_ENV_HINTS"] = "1"
    return env


# (image, disk required in GB, description) for Docker installs
IMAGES = {
    "delft": (
        "grobid/grobid:0.8.2-full",
        12.0,
        "highest accuracy (DeLFT deep-learning models)",
    ),
    "crf": (
        "grobid/grobid:0.8.2-crf",
        0.5,
        "fastest, lower accuracy (CRF only)",
    ),
}

# Disk required, in GB, for each install path (image + runtime overhead)
DISK_REQUIRED_GB = {
    "delft": 14.0,    # 12 GB image + ~2 GB working space
    "crf": 1.5,       # 500 MB image + headroom
    "source": 3.0,    # gradle deps + grobid-home models
    "colima": 30.0,   # Colima VM disk we provision
    "docker-desktop": 5.0,   # Docker.app + initial qcow allocation
}


def _fmt_gb(n: float) -> str:
    return f"{n:.1f} GB" if n >= 1.0 else f"{int(n * 1024)} MB"


def _disk_free_gb(path: Path = Path.home()) -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return float("inf")


def _check_disk(required_gb: float, label: str) -> bool:
    """Show free vs. required disk. Returns False if there's not enough."""
    free = _disk_free_gb()
    click.echo(f"Disk space: {free:.1f} GB free, ~{required_gb:.1f} GB needed for {label}.")
    if free < required_gb:
        click.secho(
            f"WARNING: only {free:.1f} GB free — {label} needs ~{required_gb:.1f} GB.",
            fg="red",
        )
        return False
    return True

# Source-install constants. GROBID is cloned into the pdf4llm repo root
# (sibling of the pdf4llm/ package) and listed in .gitignore. With an editable
# install (`pip install -e .`) this puts it at /path/to/pdf4llm/grobid.
GROBID_REPO = "https://github.com/kermitt2/grobid.git"
GROBID_VERSION_TAG = "0.8.2"
GROBID_USER_DIR = Path(__file__).resolve().parent.parent / "grobid"

DOCKER_COMPOSE = Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml"
GROBID_URL = "http://127.0.0.1:8070"


# ----------------------------- env checks -----------------------------

def check_docker() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def check_brew() -> bool:
    return shutil.which("brew") is not None


def _bootstrap_via_docker_desktop(env: dict) -> bool:
    """Fall back to Docker Desktop (works under Rosetta / when Colima can't run).

    Installs the cask, launches Docker.app, and waits for the daemon. The user
    may see one password prompt the first time Docker.app installs its helper.
    """
    desktop_app = Path("/Applications/Docker.app")
    if not desktop_app.exists():
        click.echo("Installing Docker Desktop via Homebrew cask...")
        try:
            subprocess.run(
                ["brew", "install", "--cask", "docker"],
                check=True, env=env,
            )
        except subprocess.CalledProcessError as e:
            click.secho(f"brew install --cask docker failed: {e}", fg="red")
            return False

    click.echo("Launching Docker Desktop...")
    click.secho(
        "  (Docker.app may prompt for your password the first time to install "
        "its privileged helper.)",
        fg="cyan",
    )
    try:
        subprocess.run(["open", "-a", "Docker"], check=True)
    except subprocess.CalledProcessError as e:
        click.secho(f"open -a Docker failed: {e}", fg="red")
        return False

    click.echo("Waiting for Docker daemon to become ready (up to 4 min — first launch is slow)...", nl=False)
    for _ in range(120):  # 120 × 2s = 240s
        if check_docker():
            click.secho(" OK", fg="green")
            return True
        click.echo(".", nl=False)
        time.sleep(2)
    click.secho(" TIMEOUT", fg="red")
    click.echo(
        "Docker Desktop is installed but the daemon isn't responding yet. "
        "Open Docker.app from /Applications, accept the EULA if shown, then re-run pdf4llm."
    )
    return False


def bootstrap_docker_macos() -> bool:
    """Install and start Docker Desktop on macOS.

    We use Docker Desktop unconditionally — Colima is faster but breaks under
    Rosetta and with non-native Homebrew prefixes, and the goal is a clean
    install that just works on any Mac.
    """
    if not check_brew():
        click.secho("Homebrew is required for automatic Docker install.", fg="red")
        click.echo("Install Homebrew first (one command, takes ~1 min):")
        click.echo("  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        click.echo("Then re-run pdf4llm.")
        return False

    return _bootstrap_via_docker_desktop(_brew_env())


def bootstrap_docker_linux() -> bool:
    """Best-effort Docker install on Linux. Requires sudo."""
    if shutil.which("apt-get"):
        cmds = [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "docker.io"],
            ["sudo", "systemctl", "enable", "--now", "docker"],
        ]
    elif shutil.which("dnf"):
        cmds = [
            ["sudo", "dnf", "install", "-y", "docker"],
            ["sudo", "systemctl", "enable", "--now", "docker"],
        ]
    else:
        click.secho(
            "Automatic Docker install on Linux currently supports apt-get and dnf only.",
            fg="red",
        )
        click.echo("Manual install: https://docs.docker.com/engine/install/")
        return False

    for cmd in cmds:
        click.echo(f"  $ {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            click.secho(f"Command failed: {e}", fg="red")
            return False

    if check_docker():
        click.secho("✓ Docker is ready.", fg="green")
        return True
    click.secho(
        "Docker installed but the daemon isn't reachable yet — "
        "you may need to log out / back in so your user is in the `docker` group.",
        fg="yellow",
    )
    return False


def bootstrap_docker() -> bool:
    """Install Docker for the current platform, then verify it's reachable."""
    system = platform.system()
    if system == "Darwin":
        return bootstrap_docker_macos()
    if system == "Linux":
        return bootstrap_docker_linux()
    click.secho(f"Automatic Docker install not supported on {system}.", fg="red")
    click.echo("Manual install: https://www.docker.com/products/docker-desktop")
    return False


def check_java() -> bool:
    if shutil.which("java") is None:
        return False
    try:
        r = subprocess.run(
            ["java", "-version"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def check_git() -> bool:
    return shutil.which("git") is not None


def image_present(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


# Rough estimate of how many sequence-labelling models DeLFT GROBID loads.
# Used only to render an "N/TOTAL" hint in the progress line; if more are
# observed the display falls back to "N/N".
_DELFT_EXPECTED_MODELS = 12

# Patterns we count as "another model started loading" or "another model done".
# Tuple-of-strings (not one regex) so a minor GROBID log change doesn't silently
# zero the counter — at least one pattern usually still matches.
_MODEL_LOAD_PATTERNS = (
    "loading the sequence labelling model",
    "loading sequence labelling model",
    "initializing model",
)


def _container_for_mode(mode: str) -> str:
    """Compose service name → expected container name. Matches docker-compose.yml."""
    return {"delft": "grobid-delft", "crf": "grobid-crf"}.get(mode, "")


def _start_log_tail(container: str):
    """Spawn `docker logs -f --tail 0 <container>` in the background.

    Returns the subprocess.Popen, or None if Docker isn't reachable or the
    container can't be found. Stdout is captured line-buffered for `select`.
    """
    if not container:
        return None
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "--tail", "0", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        return proc
    except (OSError, subprocess.SubprocessError):
        return None


def _drain_log_lines(proc, max_chunk: float = 0.1):
    """Read whatever's currently available from the log tail without blocking.

    Yields decoded log lines until the pipe goes idle. Spends at most
    ``max_chunk`` seconds in select waits per call so the polling loop stays
    responsive.
    """
    import select
    if proc is None or proc.stdout is None:
        return
    while True:
        r, _, _ = select.select([proc.stdout], [], [], max_chunk)
        if not r:
            return
        line = proc.stdout.readline()
        if not line:
            return
        yield line.rstrip()


def _wait_for_grobid(timeout: int = 120, mode: str = None) -> bool:
    """Poll GROBID until ready or timeout. Renders a live progress bar with an
    inline "currently loading: X (N/total)" line tailed from `docker logs`.

    Falls back to the bar-only display if Docker logs aren't reachable
    (e.g. source-built GROBID, or `docker` not on PATH).
    """
    import time
    import requests

    container = _container_for_mode(mode) if mode else ""
    log_proc = _start_log_tail(container) if container else None
    loaded_models = set()
    current_task = "starting JVM..."

    start = time.time()
    deadline = start + timeout
    click.echo(
        f"Waiting for GROBID to load (DeLFT cold-start takes ~2-3 min; "
        f"timeout {timeout}s):"
    )
    # Reserve a second line for the model-loading status so we can rewrite
    # both lines together. \n now → ANSI cursor-up later.
    click.echo("")
    sys.stdout.flush()

    first_render = True
    try:
        while time.time() < deadline:
            try:
                r = requests.get(f"{GROBID_URL}/api/isalive", timeout=5)
                if r.status_code == 200:
                    elapsed = int(time.time() - start)
                    # Clear both reserved lines, print the success line.
                    sys.stdout.write("\033[1A\r\033[2K\033[1A\r\033[2K")
                    sys.stdout.flush()
                    click.echo(f"  ✓ GROBID ready after {elapsed}s")
                    return True
            except requests.RequestException:
                pass

            # Consume any new log lines and extract task names.
            for line in _drain_log_lines(log_proc):
                low = line.lower()
                # Detect a new model load starting:
                for pat in _MODEL_LOAD_PATTERNS:
                    if pat in low:
                        # Pull the model name out of the line for display.
                        # Typical form: "... model for header (CRF)". Take the
                        # last word-ish token after "for" if present.
                        if " for " in low:
                            tail = line.split(" for ", 1)[1].strip()
                            tail = tail.split("(", 1)[0].strip().rstrip(":.,")
                            if tail:
                                current_task = tail
                        loaded_models.add(current_task)
                        break
                else:
                    # Detect a model load completing:
                    if "loaded" in low and ("model" in low or "labelling" in low):
                        loaded_models.add(current_task)

            elapsed = int(time.time() - start)
            remaining = max(0, int(deadline - time.time()))
            bar_len = 30
            filled = min(bar_len, int(bar_len * elapsed / timeout))
            bar = "█" * filled + "░" * (bar_len - filled)
            n = len(loaded_models)
            total = max(_DELFT_EXPECTED_MODELS, n)
            status = f"loading model: {current_task} ({n}/{total})"
            if not log_proc:
                status = "(docker logs unavailable — bar only)"

            # Rewrite the two reserved lines: cursor up 1 → clear → bar →
            # newline → clear → status. ESC[2K clears the whole line.
            if first_render:
                # On the very first iteration the cursor is just past the
                # reserved blank line; jump up one line then proceed normally.
                sys.stdout.write("\033[1A")
                first_render = False
            sys.stdout.write(
                "\r\033[2K"
                f"  [{bar}] {elapsed:>3}s elapsed, ~{remaining:>3}s until timeout"
                "\n\033[2K"
                f"  {status}"
                "\033[1A"  # cursor back up to the bar line for next iteration
            )
            sys.stdout.flush()
            time.sleep(2)
    finally:
        if log_proc is not None:
            try:
                log_proc.terminate()
                log_proc.wait(timeout=2)
            except (subprocess.SubprocessError, OSError):
                try:
                    log_proc.kill()
                except OSError:
                    pass

    # Move cursor down past the reserved status line so the timeout message
    # doesn't overwrite the last bar render.
    sys.stdout.write("\n")
    click.secho(
        f"  ✗ TIMEOUT after {timeout}s — last seen: {current_task}",
        fg="red",
    )
    return False


# --------------------------- Docker resource check ---------------------------

def _docker_memory_gb() -> float | None:
    """Read Docker Desktop's currently allocated RAM in GB. None if unknown."""
    try:
        r = subprocess.run(
            ["docker", "system", "info", "--format", "{{json .}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    try:
        import json
        info = json.loads(r.stdout)
        return int(info.get("MemTotal") or 0) / (1024 ** 3)
    except (ValueError, TypeError):
        return None


def _docker_desktop_settings_path() -> Path | None:
    """Location of Docker Desktop's settings JSON on macOS. None if not found."""
    if platform.system() != "Darwin":
        return None
    p = Path.home() / "Library/Group Containers/group.com.docker/settings.json"
    return p if p.exists() else None


def _bump_docker_ram_macos(target_gb: int = 10) -> bool:
    """Edit Docker Desktop's settings.json to allocate ``target_gb`` of RAM,
    then bounce Docker Desktop so the new allocation takes effect.

    Returns True on success. Backs up the original settings file first.
    """
    import json
    import time as _time

    settings = _docker_desktop_settings_path()
    if settings is None:
        return False

    # 1. Edit settings.json (with backup).
    try:
        data = json.loads(settings.read_text())
    except (OSError, ValueError) as e:
        click.secho(f"Could not read Docker settings ({e}).", fg="red")
        return False

    old_mib = data.get("memoryMiB")
    target_mib = target_gb * 1024
    if isinstance(old_mib, int) and old_mib >= target_mib:
        return True  # Already enough — nothing to do.

    backup = settings.with_suffix(".json.pdf4llm-backup")
    try:
        if not backup.exists():
            backup.write_bytes(settings.read_bytes())
        data["memoryMiB"] = target_mib
        settings.write_text(json.dumps(data, indent=2))
    except OSError as e:
        click.secho(f"Could not write Docker settings ({e}).", fg="red")
        return False

    click.echo(f"  edited Docker settings: memoryMiB {old_mib} → {target_mib}")
    click.echo(f"  backup at {backup}")

    # 2. Restart Docker Desktop.
    click.echo("  quitting Docker Desktop...")
    subprocess.run(["osascript", "-e", 'quit app "Docker"'],
                   capture_output=True, check=False)
    # Wait for daemon to stop
    for _ in range(30):
        if not check_docker():
            break
        _time.sleep(1)

    click.echo("  relaunching Docker Desktop...")
    subprocess.run(["open", "-a", "Docker"], capture_output=True, check=False)
    # Wait for daemon to come back (up to 90 s)
    for _ in range(45):
        if check_docker():
            break
        _time.sleep(2)
    if not check_docker():
        click.secho(
            "  Docker did not come back online — please launch Docker.app manually.",
            fg="red",
        )
        return False

    # 3. Verify the new allocation actually stuck.
    new_gb = _docker_memory_gb()
    if new_gb is None:
        return True  # Couldn't probe — assume OK.
    if new_gb < target_gb - 0.5:
        click.secho(
            f"  Docker came back with only {new_gb:.1f} GB (expected ≥ {target_gb}). "
            "Settings change may have been overridden by Docker Desktop.",
            fg="yellow",
        )
        return False
    click.secho(f"  ✓ Docker Desktop now has {new_gb:.1f} GB allocated.", fg="green")
    return True


def _check_docker_resources(min_memory_gb: float = 8.0,
                             auto_bump: bool = True,
                             auto_bump_target_gb: int = 10) -> tuple:
    """Verify Docker Desktop is allocated enough RAM for DeLFT GROBID.

    If under-allocated AND we're on macOS AND ``auto_bump`` is True, attempt
    to edit settings.json and bounce Docker Desktop automatically. Falls back
    to a clear error with manual fix instructions if the auto-bump fails.

    Returns (ok, message). On True the message is empty.
    """
    mem_gb = _docker_memory_gb()
    if mem_gb is None:
        return True, ""  # Can't tell — don't block.
    if mem_gb >= min_memory_gb:
        return True, ""

    if auto_bump and platform.system() == "Darwin" and _docker_desktop_settings_path():
        click.secho(
            f"\nDocker Desktop has only {mem_gb:.1f} GB RAM allocated; "
            f"GROBID DeLFT needs ≥ {min_memory_gb:.0f} GB.",
            fg="yellow",
        )
        click.echo(f"Auto-bumping Docker Desktop RAM to {auto_bump_target_gb} GB...")
        if _bump_docker_ram_macos(target_gb=auto_bump_target_gb):
            return True, ""
        # Auto-bump failed — fall through to manual-instructions path.

    msg = (
        f"Docker Desktop has only {mem_gb:.1f} GB RAM allocated, but "
        f"GROBID DeLFT needs ≥ {min_memory_gb:.0f} GB (its JVM runs with -Xmx6g "
        "and needs headroom).\n\n"
        "Fix this in Docker Desktop:\n"
        f"  Settings → Resources → Memory → drag to {min_memory_gb:.0f} GB "
        "(or higher) → Apply & Restart\n\n"
        "Then re-run pdf4llm. Or use:\n"
        "  pdf4llm-install-grobid --mode crf\n"
        "  (skips DeLFT, ~500 MB image, ~30 s start, lower reference accuracy)"
    )
    return False, msg


# --------------------------- Docker install ---------------------------

def install_docker(mode: str = "delft", start: bool = True) -> bool:
    if mode not in IMAGES:
        click.secho(f"Unknown mode: {mode!r}", fg="red")
        return False

    image, size_gb, desc = IMAGES[mode]

    if not check_docker():
        click.secho("Docker is not installed or not running.", fg="red")
        click.echo("Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        return False

    if not image_present(image):
        _check_disk(DISK_REQUIRED_GB[mode], f"GROBID {mode} image")
        click.echo(f"Pulling {image} ({_fmt_gb(size_gb)}, {desc})...")
        try:
            subprocess.run(["docker", "pull", image], check=True)
        except subprocess.CalledProcessError as e:
            click.secho(f"docker pull failed: {e}", fg="red")
            return False
    else:
        click.echo(f"Image {image} already present.")

    if not start:
        return True

    if not DOCKER_COMPOSE.exists():
        click.secho(f"docker-compose.yml not found at {DOCKER_COMPOSE}", fg="red")
        return False

    # Pre-flight: refuse to start DeLFT if Docker Desktop doesn't have enough RAM
    # allocated. Without this, the user waits 5 minutes for a doomed cold-start.
    # (CRF mode runs with -Xmx4g so we only enforce the full 8 GB bar for DeLFT.)
    if mode == "delft":
        ok, msg = _check_docker_resources(min_memory_gb=8.0)
        if not ok:
            click.secho("\n" + msg, fg="red")
            return False

    click.echo(f"Starting GROBID ({mode} mode)...")
    subprocess.run(
        ["docker", "compose", "-f", str(DOCKER_COMPOSE), "down"],
        capture_output=True, check=False,
    )
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_COMPOSE),
             "--profile", mode, "up", "-d"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.secho(f"Failed to start GROBID: {e}", fg="red")
        return False

    if not _wait_for_grobid(timeout=300 if mode == "delft" else 120, mode=mode):
        click.secho("GROBID started but did not become ready.", fg="red")
        click.echo(f"Check logs: docker compose -f {DOCKER_COMPOSE} logs")
        return False

    click.secho(f"✓ GROBID is running at {GROBID_URL}", fg="green")
    return True


# --------------------------- source install ---------------------------

def install_source(target_dir: Path = GROBID_USER_DIR, start: bool = True) -> bool:
    """Clone GROBID and build it with the Gradle wrapper.

    Builds CRF-only mode (no DeLFT, no Python venv). The user can later add
    DeLFT support by following GROBID's docs.
    """
    if not check_java():
        click.secho("Java is not installed.", fg="red")
        click.echo("Install Java 11 or 17 first:")
        click.echo("  macOS:  brew install openjdk@17")
        click.echo("  Linux:  sudo apt-get install openjdk-17-jdk")
        return False

    if not check_git():
        click.secho("git is not installed.", fg="red")
        click.echo("Install git first (macOS: `xcode-select --install` or `brew install git`).")
        return False

    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    _check_disk(DISK_REQUIRED_GB["source"], "GROBID source build")

    if not (target_dir / ".git").exists():
        click.echo(f"Cloning GROBID {GROBID_VERSION_TAG} into {target_dir}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "--branch", GROBID_VERSION_TAG, GROBID_REPO, str(target_dir)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            click.secho(f"git clone failed: {e}", fg="red")
            return False
    else:
        click.echo(f"GROBID source already present at {target_dir}; skipping clone.")

    gradlew = target_dir / "gradlew"
    if not gradlew.exists():
        click.secho(f"gradlew not found in {target_dir}", fg="red")
        return False
    gradlew.chmod(0o755)

    click.echo("Building GROBID (this can take 10–20 minutes on first run)...")
    try:
        subprocess.run(
            [str(gradlew), "clean", "install"],
            cwd=str(target_dir), check=True,
        )
    except subprocess.CalledProcessError as e:
        click.secho(f"GROBID build failed: {e}", fg="red")
        return False

    click.secho(f"✓ GROBID built at {target_dir}", fg="green")
    click.echo("")
    click.echo("To use this install, point pdf4llm at it with:")
    click.echo(f"  --grobid-home {target_dir}")
    click.echo("or pdf4llm will find it automatically (this path is checked by default).")

    if not start:
        return True

    click.echo("Starting GROBID (CRF mode)...")
    proc = subprocess.Popen(
        [str(gradlew), "run"],
        cwd=str(target_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if not _wait_for_grobid(timeout=180):
        click.secho("GROBID failed to start.", fg="red")
        try:
            proc.terminate()
        except Exception:
            pass
        return False

    click.secho(f"✓ GROBID is running at {GROBID_URL}", fg="green")
    click.echo("(GROBID is running in the background; pdf4llm will reuse it.)")
    return True


# --------------------------- interactive ---------------------------

def prompt_and_install() -> bool:
    """Interactive prompt used by the CLI when GROBID isn't responding.

    Distinguishes four states and tailors the message:
      • Docker not installed
      • GROBID image not pulled
      • Image pulled but container not running (or crashed previously)
      • Image pulled, container running, but isalive failed (loading or stuck)

    Returns True if GROBID is alive and reachable when the function returns.
    """
    docker_installed = check_docker()
    image = IMAGES["delft"][0]
    image_present_flag = docker_installed and image_present(image)
    container_state = _grobid_container_state(docker_installed)

    # Tailored headline + action verb
    if not docker_installed:
        click.secho("\nGROBID isn't available — Docker isn't installed.", fg="yellow", bold=True)
        action = "Install Docker + GROBID DeLFT now?"
    elif not image_present_flag:
        click.secho("\nGROBID DeLFT image isn't installed.", fg="yellow", bold=True)
        action = "Pull and start GROBID DeLFT now?"
    elif container_state and container_state[0] == "exited":
        exit_code = container_state[1]
        hint = ""
        if exit_code in (134, 137, 139):
            # 134=SIGABRT (often JVM OOM-abort), 137=OOM-killed, 139=segfault
            hint = (
                "  (exit 134/137 usually means the JVM ran out of memory — "
                "bump Docker Desktop RAM to ≥ 10 GB)"
            )
        click.secho(
            f"\nGROBID DeLFT image is installed, but the container exited (code {exit_code}).",
            fg="yellow", bold=True,
        )
        if hint:
            click.echo(hint)
        action = "Restart GROBID DeLFT now?"
    elif container_state and container_state[0] == "running":
        click.secho(
            "\nGROBID DeLFT container is running but isn't responding on the API yet.",
            fg="yellow", bold=True,
        )
        click.echo(
            "  Either it's still loading models (cold-start is 2-3 min) or it's stuck."
        )
        action = "Restart GROBID DeLFT and wait again?"
    else:
        click.secho(
            "\nGROBID DeLFT image is installed but no container is running.",
            fg="yellow", bold=True,
        )
        action = "Start GROBID DeLFT now?"

    click.echo("GROBID DeLFT powers high-accuracy reference parsing and header extraction.")
    click.echo("")

    free = _disk_free_gb()
    # If the image is already pulled we don't need ~14 GB free — just headroom.
    if image_present_flag:
        needed = 1.0       # just enough working space
        needed_with_buffer = 2.0
        click.echo(f"Disk free on $HOME: {free:.1f} GB (image already pulled, no large download)")
    elif docker_installed:
        needed = DISK_REQUIRED_GB["delft"]
        needed_with_buffer = needed + 2.0
        click.echo(f"Disk free on $HOME: {free:.1f} GB")
        click.echo(f"Disk needed:        ~{needed:.0f} GB (GROBID DeLFT image)")
    else:
        needed = DISK_REQUIRED_GB["delft"] + DISK_REQUIRED_GB["docker-desktop"]
        needed_with_buffer = needed + 2.0
        click.echo(f"Disk free on $HOME: {free:.1f} GB")
        click.echo(
            f"Disk needed:        ~{needed:.0f} GB "
            f"(Docker Desktop ~{DISK_REQUIRED_GB['docker-desktop']:.0f} GB + "
            f"GROBID DeLFT ~{DISK_REQUIRED_GB['delft']:.0f} GB)"
        )
    click.echo("")

    # Hard precheck: refuse if there isn't enough disk.
    if free < needed_with_buffer:
        click.secho(
            f"Not enough free disk (~{needed_with_buffer:.1f} GB needed, "
            f"{free:.1f} GB free).",
            fg="red",
        )
        click.secho("Falling back to docling-only extraction.", fg="yellow")
        return False

    if not click.confirm(action, default=True):
        return False

    if not docker_installed:
        click.echo("")
        click.echo("Installing Docker Desktop first...")
        if not bootstrap_docker():
            click.secho("Docker install failed — cannot proceed with GROBID.", fg="red")
            return False
        click.echo("")

    return install_docker(mode="delft", start=True)


def _grobid_container_state(docker_ok: bool) -> tuple:
    """Inspect existing GROBID containers; return (state, exit_code|None) or None.

    state ∈ {"running", "exited", "created", "paused", ...}. Looks for the
    well-known service container names from docker-compose.yml.
    """
    if not docker_ok:
        return None
    for name in ("grobid-delft", "grobid-crf"):
        try:
            r = subprocess.run(
                ["docker", "inspect", "--format",
                 "{{.State.Status}}|{{.State.ExitCode}}", name],
                capture_output=True, text=True, timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if r.returncode != 0:
            continue
        state, _, code = r.stdout.strip().partition("|")
        try:
            exit_code = int(code) if code else None
        except ValueError:
            exit_code = None
        return (state, exit_code)
    return None


# --------------------------- CLI entry point ---------------------------

@click.command("pdf4llm-install-grobid")
@click.option(
    "--mode",
    type=click.Choice(["source", "delft", "crf"]),
    default="source",
    show_default=True,
    help="Install path: source (build from source, no Docker), "
         "delft (Docker ~12 GB), or crf (Docker ~500 MB).",
)
@click.option(
    "--target",
    type=click.Path(),
    default=str(GROBID_USER_DIR),
    show_default=True,
    help="Where to clone GROBID (source mode only).",
)
@click.option(
    "--no-start", is_flag=True,
    help="Install only; do not start GROBID afterwards.",
)
def main(mode, target, no_start):
    """Install GROBID for pdf4llm (source build or Docker image)."""
    if mode == "source":
        ok = install_source(target_dir=Path(target), start=not no_start)
    else:
        # Bootstrap Docker if it's missing (Colima on macOS, apt/dnf on Linux)
        if not check_docker():
            click.secho("Docker is not installed — attempting auto-install.", fg="yellow")
            if not bootstrap_docker():
                sys.exit(1)
        image, size_gb, desc = IMAGES[mode]
        click.echo(f"This will download {image} ({_fmt_gb(size_gb)}, {desc}).")
        _check_disk(DISK_REQUIRED_GB[mode], f"GROBID {mode}")
        ok = install_docker(mode=mode, start=not no_start)
    sys.exit(0 if ok else 1)


# --------------------------- docling install/repair ---------------------------

# Disk needed for docling + ML model cache + numpy<2 reinstall
DOCLING_DISK_GB = 5.0


def _docling_importable() -> tuple[bool, str]:
    """Return (True, '') if docling imports cleanly, else (False, error)."""
    try:
        import docling  # noqa: F401
        from docling.document_converter import DocumentConverter  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)


def ensure_docling() -> bool:
    """Make sure docling is usable. If not, prompt to install/repair.

    Cases handled:
      • Docling already importable → no-op, return True.
      • Docling missing → install docling.
      • Numpy 1.x / 2.x ABI mismatch → force-reinstall the C-extension stragglers
        (numpy, pandas, pyarrow) so they all come from PyPI wheels with a
        consistent ABI, then re-exec Python (C extensions can't be unloaded
        in-process, so the running interpreter has to be replaced).

    On any successful install this function does NOT return — it re-execs the
    same command. Returns False only if the user declines, disk is short, or
    the install itself fails.
    """
    ok, err = _docling_importable()
    if ok:
        return True

    # Guard: never run `pip install` outside of a venv. Doing so would mutate
    # the user's base Python (e.g. anaconda) and create cross-package conflicts
    # — that's exactly the failure mode this whole launcher/venv setup exists
    # to prevent. Direct the user to the venv path instead.
    in_venv = (
        os.environ.get("PDF4LLM_VENV_ACTIVE") == "1"
        or os.environ.get("VIRTUAL_ENV")
    )
    if not in_venv:
        click.secho("\ndocling cannot be imported in this Python environment.", fg="red", bold=True)
        click.echo(f"Reason: {err}")
        click.echo("")
        click.echo("You're running outside the pdf4llm-managed venv, so I won't")
        click.echo("`pip install` here — that would just pollute your base env.")
        click.echo("")
        click.echo("Fix:  pdf4llm --reinstall-venv yourfile.pdf")
        click.echo("      (creates / refreshes the isolated venv, then re-runs)")
        return False

    is_numpy_abi = (
        "numpy" in err.lower()
        and ("abi" in err.lower()
             or "multiarray" in err.lower()
             or "compiled using numpy" in err.lower())
    )

    click.secho("\ndocling cannot be imported.", fg="yellow", bold=True)
    if is_numpy_abi:
        click.echo("Reason: numpy 1.x / 2.x ABI mismatch — pandas/pyarrow in this env")
        click.echo("        were compiled against a different numpy than the one loaded.")
        click.echo("Fix:    reinstall numpy + pandas + pyarrow as a consistent set.")
    else:
        click.echo(f"Reason: {err}")
        click.echo("Fix:    pip install docling")

    free = _disk_free_gb()
    click.echo(f"Disk free:   {free:.1f} GB")
    click.echo(f"Disk needed: ~{DOCLING_DISK_GB:.0f} GB (docling + ML model cache on first run)")

    if free < DOCLING_DISK_GB:
        click.secho(
            f"Not enough free disk (~{DOCLING_DISK_GB:.0f} GB needed, {free:.1f} GB free).",
            fg="red",
        )
        return False

    if not click.confirm("Install / repair docling now?", default=True):
        return False

    base = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if is_numpy_abi:
        # Force-reinstall the C-extension stragglers so they all come from
        # PyPI built against a consistent numpy ABI. Let pip's resolver pick
        # compatible versions — do NOT pin numpy here.
        pip_args = base + ["--force-reinstall", "--no-cache-dir",
                           "numpy", "pandas", "pyarrow", "docling"]
    else:
        pip_args = base + ["docling"]

    click.echo("Running: " + " ".join(pip_args))
    try:
        subprocess.run(pip_args, check=True)
    except subprocess.CalledProcessError as e:
        click.secho(f"pip install failed: {e}", fg="red")
        return False

    # C-extension modules already loaded in this interpreter (numpy, pandas, etc.)
    # cannot be replaced in-place — we have to re-exec Python with the same argv
    # so the next process starts with a clean module table.
    click.secho(
        "\n✓ Reinstall finished. Restarting pdf4llm to pick up the new "
        "numpy / pandas / pyarrow...\n",
        fg="green",
    )
    sys.stdout.flush()

    # Skip the GROBID prompt in the re-exec'd process — the user already chose
    # docling-only when they declined GROBID before reaching this function.
    child_env = os.environ.copy()
    child_env["PDF4LLM_FORCE_DOCLING"] = "1"
    try:
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "pdf4llm.cli", *sys.argv[1:]],
            child_env,
        )
    except OSError as e:
        click.secho(
            f"Could not re-exec ({e}). Install succeeded — please re-run your command.",
            fg="yellow",
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
