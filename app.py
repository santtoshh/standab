"""Flask web app that runs the Standab analysis notebook server-side.

Users upload one or more ride CSVs through the browser; the notebook is executed
with those CSVs injected as parameters and the resulting `maps/output.html` is
streamed back as a download. No Python source is ever sent to the client.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("IPYTHONDIR", str(Path(tempfile.gettempdir()) / "standab_ipython"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "standab_jupyter"))

import nbformat
from flask import Flask, abort, render_template, request, send_file
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Standab Data Analysis Tool.ipynb"

EXECUTION_TIMEOUT_SEC = int(os.environ.get("STANDAB_EXEC_TIMEOUT", "900"))

app = Flask(__name__)
# No hard upload size limit. The practical ceiling is the proxy/platform
# request body limit (e.g. Railway's edge proxy) and the server's memory.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("standab-web")


def _sanitize_operator_key(raw: str, fallback_index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (raw or "").strip()).strip("_-")
    if not cleaned:
        cleaned = f"OP{fallback_index + 1}"
    return cleaned.upper()[:32]


def _sanitize_stem(raw: str) -> str:
    """Filename-safe version of the uploaded file's stem (no extension)."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (raw or "").strip()).strip("_-")
    return cleaned[:60]


def _build_download_name(original_filenames: list[str]) -> str:
    """Build the output HTML filename from the uploaded data file stems + timestamp.

    Single upload  -> ``<stem>_<YYYY-MM-DD_HH-MM>.html``
    Multiple files -> ``<stem1>_<stem2>_<YYYY-MM-DD_HH-MM>.html``
                       (capped at 3 stems; "+N" suffix if more)
    """
    stems = [s for s in (_sanitize_stem(Path(n or "").stem) for n in original_filenames) if s]
    if not stems:
        stems = ["standab-analysis"]

    if len(stems) > 3:
        joined = "_".join(stems[:3]) + f"+{len(stems) - 3}"
    else:
        joined = "_".join(stems)

    if len(joined) > 120:
        joined = joined[:120].rstrip("_-")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{joined}_{timestamp}.html"


_BOOT_SRC_TEMPLATE = (
    "# [standab-web] boot cell: set sys.path + cwd before any notebook code runs\n"
    "import os, sys\n"
    "sys.path.insert(0, {project_root!r})\n"
    "os.chdir({workdir!r})\n"
)


def _write_dummy_swaps_csv(workdir: Path) -> Path:
    path = workdir / "_swaps_dummy.csv"
    path.write_text("col\n0\n", encoding="utf-8")
    return path


def _find_rides_paths_cell(nb: nbformat.NotebookNode) -> int | None:
    """Return the index of the first cell that assigns RIDES_CSV_PATHS."""
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if re.search(r"^\s*RIDES_CSV_PATHS\s*=", src, flags=re.MULTILINE):
            return idx
    return None


def _run_notebook(rides_paths: dict[str, str], workdir: Path) -> Path:
    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    swaps_dummy = _write_dummy_swaps_csv(workdir)

    boot_src = _BOOT_SRC_TEMPLATE.format(
        project_root=str(PROJECT_ROOT),
        workdir=str(workdir),
    )

    # Boot cell at index 0 handles sys.path + cwd before any imports.
    nb.cells.insert(0, nbformat.v4.new_code_cell(boot_src))

    # The override must run AFTER the notebook's own RIDES_CSV_PATHS assignment
    # but BEFORE the data is actually loaded. We replace the original assignment
    # cell's body entirely so our paths win regardless of what the notebook does.
    rides_cell_idx = _find_rides_paths_cell(nb)
    if rides_cell_idx is None:
        raise RuntimeError(
            "Could not find the RIDES_CSV_PATHS assignment cell in the notebook."
        )
    original = nb.cells[rides_cell_idx].get("source", "")
    if isinstance(original, list):
        original = "".join(original)

    # Inline-replace each dev-only hardcoded constant with our values so that
    # any downstream validation in the same cell still sees the overrides.
    patched = re.sub(
        r"RIDES_CSV_PATHS\s*=\s*\{[\s\S]*?\}",
        f"RIDES_CSV_PATHS = {rides_paths!r}  # [standab-web] injected",
        original,
        count=1,
    )
    patched = re.sub(
        r"^\s*SWAPS_CSV_PATH\s*=.*$",
        f"SWAPS_CSV_PATH = {str(swaps_dummy)!r}  # [standab-web] injected",
        patched,
        count=1,
        flags=re.MULTILINE,
    )
    patched = re.sub(
        r"^\s*PARKING_HUBS_DBF_PATH\s*=.*$",
        "PARKING_HUBS_DBF_PATH = None  # [standab-web] injected",
        patched,
        count=1,
        flags=re.MULTILINE,
    )
    patched = re.sub(
        r"^\s*PARKING_HUBS_UI_LABEL\s*=.*$",
        "PARKING_HUBS_UI_LABEL = ''  # [standab-web] injected",
        patched,
        count=1,
        flags=re.MULTILINE,
    )
    nb.cells[rides_cell_idx] = nbformat.v4.new_code_cell(patched)

    client = NotebookClient(
        nb,
        timeout=EXECUTION_TIMEOUT_SEC,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
        allow_errors=False,
    )
    client.execute()

    output_html = workdir / "maps" / "output.html"
    if not output_html.exists():
        raise RuntimeError(
            f"Notebook completed but {output_html} was not produced."
        )
    return output_html


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


@app.route("/generate", methods=["POST"])
def generate():
    names = request.form.getlist("operator_name")
    files = request.files.getlist("operator_csv")

    pairs: list[tuple[str, "FileStorage"]] = []  # type: ignore[name-defined]
    for idx, storage in enumerate(files):
        if not storage or not storage.filename:
            continue
        raw_name = names[idx] if idx < len(names) else ""
        if not raw_name.strip():
            raw_name = Path(storage.filename).stem
        key = _sanitize_operator_key(raw_name, idx)
        pairs.append((key, storage))

    if not pairs:
        abort(400, "Please upload at least one CSV file.")

    seen: set[str] = set()
    deduped: list[tuple[str, "FileStorage"]] = []  # type: ignore[name-defined]
    for key, storage in pairs:
        final = key
        bump = 2
        while final in seen:
            final = f"{key}_{bump}"
            bump += 1
        seen.add(final)
        deduped.append((final, storage))

    job_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f"standab-{job_id}-"))
    try:
        uploads_dir = workdir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        rides_paths: dict[str, str] = {}
        for key, storage in deduped:
            safe = secure_filename(storage.filename) or f"{key}.csv"
            dest = uploads_dir / f"{key}__{safe}"
            storage.save(str(dest))
            rides_paths[key] = str(dest)

        log.info("job=%s operators=%s workdir=%s", job_id, list(rides_paths.keys()), workdir)
        output_html = _run_notebook(rides_paths, workdir)

        with open(output_html, "rb") as fh:
            blob = fh.read()

        download_name = _build_download_name(
            [storage.filename for _key, storage in deduped]
        )
        return send_file(
            io.BytesIO(blob),
            mimetype="text/html",
            as_attachment=True,
            download_name=download_name,
        )
    except CellExecutionError as exc:
        log.exception("job=%s notebook execution failed", job_id)
        snippet = str(exc)
        if len(snippet) > 4000:
            snippet = snippet[:2000] + "\n...[truncated]...\n" + snippet[-2000:]
        return (
            render_template(
                "error.html",
                job_id=job_id,
                message="The analysis notebook raised an error while processing your data.",
                details=snippet,
            ),
            500,
        )
    except Exception as exc:
        log.exception("job=%s failed", job_id)
        return (
            render_template(
                "error.html",
                job_id=job_id,
                message="Unexpected server error while generating the report.",
                details=str(exc),
            ),
            500,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
