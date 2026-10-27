"""Flask web app that runs the Standab analysis notebook server-side.

Two-stage upload flow:

  1. ``POST /preview`` — receives one or more uploaded files (CSV or XLSX),
     saves them to a per-job workdir, peeks at the headers + first rows,
     and returns a JSON proposal so the frontend can render the
     column-mapping confirmation page.

  2. ``POST /generate`` — receives the user-confirmed mapping, materializes
     a canonical-named CSV per operator inside the same workdir, then
     executes the notebook with those canonical CSVs as input. The rendered
     ``maps/output.html`` is streamed back as a download.

No Python source is ever sent to the client.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("IPYTHONDIR", str(Path(tempfile.gettempdir()) / "standab_ipython"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "standab_jupyter"))

import nbformat
from flask import Flask, abort, jsonify, render_template, request, send_file
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from werkzeug.utils import secure_filename

from dataset_import import (
    materialize_dataset,
    materialize_parking_dataset,
    peek_dataset,
    peek_parking_dataset,
    validate_mapping,
    validate_parking_mapping,
)

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Standab Data Analysis Tool.ipynb"

EXECUTION_TIMEOUT_SEC = int(os.environ.get("STANDAB_EXEC_TIMEOUT", "900"))
JOB_TTL_SEC = int(os.environ.get("STANDAB_JOB_TTL", "3600"))
JOB_DIR_PREFIX = "standab-"

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


# ---------------------------------------------------------------------------
# Background jobs: the notebook runs in a thread so the HTTP request returns
# at once and the page polls /status/<job>. Long runs no longer hit the proxy's
# request timeout, which surfaced to users as "upstream error".
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _job_set(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def _job_get(job_id: str) -> dict:
    with JOBS_LOCK:
        return dict(JOBS.get(job_id) or {})


def _job_gc(now: float | None = None) -> None:
    cutoff = (now if now is not None else time.time()) - JOB_TTL_SEC
    with JOBS_LOCK:
        for key in [k for k, v in JOBS.items() if v.get("started", 0) < cutoff]:
            JOBS.pop(key, None)


def _sanitize_job_id(raw: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", (raw or "").strip())[:32]
    return cleaned or None


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


def _new_workdir() -> tuple[str, Path]:
    """Create a fresh per-job workdir under ``$TMPDIR`` and return (job_id, path)."""
    job_id = uuid.uuid4().hex[:12]
    workdir = Path(tempfile.mkdtemp(prefix=f"{JOB_DIR_PREFIX}{job_id}-"))
    (workdir / "uploads").mkdir(parents=True, exist_ok=True)
    return job_id, workdir


def _workdir_for(job_id: str) -> Path | None:
    """Resolve the existing workdir for a given job id, or ``None`` if missing.

    Per-job dirs are named ``<TMPDIR>/standab-<job_id>-XXXXXXXX`` where the
    suffix is generated by ``mkdtemp``. We search by prefix to recover the
    actual path between the two HTTP requests that share a job.
    """
    if not job_id:
        return None
    needle = f"{JOB_DIR_PREFIX}{job_id}-"
    tmp_root = Path(tempfile.gettempdir())
    matches = sorted(tmp_root.glob(f"{needle}*"))
    for candidate in matches:
        if candidate.is_dir():
            return candidate
    return None


def _gc_old_jobdirs(now: float | None = None) -> None:
    """Delete any ``standab-*`` workdir older than ``JOB_TTL_SEC`` seconds.

    Cheap best-effort sweep called on every request handler. Failures are
    swallowed because /tmp cleanup must never break a real request.
    """
    cutoff = (now if now is not None else time.time()) - JOB_TTL_SEC
    tmp_root = Path(tempfile.gettempdir())
    try:
        candidates = list(tmp_root.glob(f"{JOB_DIR_PREFIX}*"))
    except OSError:
        return
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            mtime = candidate.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(candidate, ignore_errors=True)
        except OSError:
            continue


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


def _run_notebook(
    rides_paths: dict[str, str],
    workdir: Path,
    parking_path: str | None = None,
    parking_label: str = "",
) -> Path:
    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    swaps_dummy = _write_dummy_swaps_csv(workdir)

    boot_src = _BOOT_SRC_TEMPLATE.format(
        project_root=str(PROJECT_ROOT),
        workdir=str(workdir),
    )

    nb.cells.insert(0, nbformat.v4.new_code_cell(boot_src))

    rides_cell_idx = _find_rides_paths_cell(nb)
    if rides_cell_idx is None:
        raise RuntimeError(
            "Could not find the RIDES_CSV_PATHS assignment cell in the notebook."
        )
    original = nb.cells[rides_cell_idx].get("source", "")
    if isinstance(original, list):
        original = "".join(original)

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
    parking_value = repr(parking_path) if parking_path else "None"
    if re.search(r"^\s*PARKING_HUBS_PATH\s*=", patched, flags=re.MULTILINE):
        patched = re.sub(
            r"^\s*PARKING_HUBS_PATH\s*=.*$",
            f"PARKING_HUBS_PATH = {parking_value}  # [standab-web] injected",
            patched,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        # Older notebooks may not declare PARKING_HUBS_PATH; append after the DBF line.
        patched = re.sub(
            r"(^\s*PARKING_HUBS_DBF_PATH\s*=.*$)",
            r"\1\n" + f"PARKING_HUBS_PATH = {parking_value}  # [standab-web] injected",
            patched,
            count=1,
            flags=re.MULTILINE,
        )
    patched = re.sub(
        r"^\s*PARKING_HUBS_UI_LABEL\s*=.*$",
        f"PARKING_HUBS_UI_LABEL = {parking_label!r}  # [standab-web] injected",
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


@app.before_request
def _sweep_old_jobs():
    _gc_old_jobdirs()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    info = {"status": "ok"}
    try:
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        info["memory_limit_mb"] = None if lim == "max" else round(int(lim) / 1e6)
        info["memory_used_mb"] = round(int(Path("/sys/fs/cgroup/memory.current").read_text().strip()) / 1e6)
    except Exception:
        pass
    with JOBS_LOCK:
        info["jobs_running"] = sum(1 for v in JOBS.values() if v.get("status") == "running")
    return jsonify(info)


@app.route("/preview", methods=["POST"])
def preview():
    """Stage 1: receive uploads, peek headers + propose column mapping."""
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
        return jsonify({"error": "Please upload at least one file."}), 400

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

    job_id, workdir = _new_workdir()
    uploads_dir = workdir / "uploads"

    response_files: list[dict] = []
    try:
        for idx, (key, storage) in enumerate(deduped):
            safe = secure_filename(storage.filename) or f"{key}.csv"
            saved_as = f"{key}__{safe}"
            dest = uploads_dir / saved_as
            storage.save(str(dest))

            try:
                peek = peek_dataset(dest)
            except Exception as exc:
                log.exception("preview: failed to peek %s", dest)
                return (
                    jsonify(
                        {
                            "error": (
                                f"Could not read file '{storage.filename}': {exc}. "
                                "Confirm it is a valid CSV or XLSX."
                            )
                        }
                    ),
                    400,
                )

            raw_label = (
                names[idx]
                if idx < len(names) and (names[idx] or "").strip()
                else Path(storage.filename).stem
            )
            response_files.append(
                {
                    "operator_key": key,
                    "operator_label": raw_label.strip() or key,
                    "original_name": storage.filename,
                    "saved_as": saved_as,
                    "is_excel": Path(saved_as).suffix.lower() in {".xlsx", ".xls"},
                    **peek,
                }
            )
    except Exception as exc:
        log.exception("preview: unexpected failure (job=%s)", job_id)
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500

    # Optional parking-hubs overlay file (single, applies to the whole analysis).
    parking_block = None
    parking_storage = request.files.get("parking_csv")
    if parking_storage and parking_storage.filename:
        try:
            safe = secure_filename(parking_storage.filename) or "parking.csv"
            saved_as = f"PARKING__{safe}"
            dest = uploads_dir / saved_as
            parking_storage.save(str(dest))
            peek = peek_parking_dataset(dest)
            parking_label_raw = (request.form.get("parking_label") or "").strip()
            parking_block = {
                "original_name": parking_storage.filename,
                "saved_as": saved_as,
                "label": parking_label_raw or Path(parking_storage.filename).stem,
                "is_excel": Path(saved_as).suffix.lower() in {".xlsx", ".xls"},
                **peek,
            }
        except Exception as exc:
            log.exception("preview: failed to peek parking file (job=%s)", job_id)
            shutil.rmtree(workdir, ignore_errors=True)
            return (
                jsonify(
                    {
                        "error": (
                            f"Could not read parking file "
                            f"'{parking_storage.filename}': {exc}. "
                            "Confirm it is a valid CSV or XLSX."
                        )
                    }
                ),
                400,
            )

    log.info(
        "preview: job=%s files=%d operators=%s parking=%s",
        job_id,
        len(response_files),
        [f["operator_key"] for f in response_files],
        bool(parking_block),
    )
    return jsonify(
        {"job_id": job_id, "files": response_files, "parking": parking_block}
    )


@app.route("/peek_sheet", methods=["POST"])
def peek_sheet():
    """Re-peek an already-uploaded file with a different sheet selection."""
    payload = request.get_json(silent=True) or {}
    job_id = _sanitize_job_id(payload.get("job_id", ""))
    saved_as = payload.get("saved_as", "")
    sheet = payload.get("sheet")

    workdir = _workdir_for(job_id) if job_id else None
    if not workdir:
        return jsonify({"error": "Job not found or expired."}), 400
    if not saved_as or "/" in saved_as or "\\" in saved_as or saved_as.startswith(".."):
        return jsonify({"error": "Invalid file reference."}), 400
    src = workdir / "uploads" / saved_as
    if not src.exists():
        return jsonify({"error": "Uploaded file is no longer available."}), 400

    kind = (payload.get("kind") or "rides").lower()
    try:
        if kind == "parking":
            peek = peek_parking_dataset(src, sheet=sheet)
        else:
            peek = peek_dataset(src, sheet=sheet)
    except Exception as exc:
        log.exception("peek_sheet: failed for %s", src)
        return jsonify({"error": f"Could not read sheet: {exc}"}), 400

    return jsonify(peek)


@app.route("/generate", methods=["POST"])
def generate():
    """Stage 2: apply confirmed mapping, run notebook, stream output.html."""
    payload_raw = request.form.get("payload")
    if payload_raw:
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError as exc:
            return _render_error(
                None,
                "Could not parse mapping payload.",
                f"{exc}",
                http_status=400,
            )
    else:
        payload = request.get_json(silent=True) or {}

    job_id = _sanitize_job_id(str(payload.get("job_id", "")))
    files_payload = payload.get("files") or []

    if not job_id or not files_payload:
        return _render_error(
            None,
            "Mapping confirmation is incomplete.",
            "Missing job_id or files. Please go back and re-upload.",
            http_status=400,
        )

    workdir = _workdir_for(job_id)
    if not workdir:
        return _render_error(
            job_id,
            "Your upload session has expired.",
            "Please go back to the upload page and try again. "
            "The server keeps uploads on disk for a limited time.",
            http_status=400,
        )

    uploads_dir = workdir / "uploads"
    rides_paths: dict[str, str] = {}
    original_filenames: list[str] = []

    try:
        seen_keys: set[str] = set()
        for entry in files_payload:
            saved_as = (entry.get("saved_as") or "").strip()
            operator_key_raw = (entry.get("operator_key") or "").strip()
            if not saved_as or not operator_key_raw:
                raise ValueError("Each file entry needs operator_key + saved_as.")
            if "/" in saved_as or "\\" in saved_as or saved_as.startswith(".."):
                raise ValueError(f"Invalid file reference: {saved_as}")
            src = uploads_dir / saved_as
            if not src.exists():
                raise ValueError(
                    f"Uploaded file '{saved_as}' is no longer available. "
                    "Please re-upload."
                )

            key = _sanitize_operator_key(operator_key_raw, len(seen_keys))
            final_key = key
            bump = 2
            while final_key in seen_keys:
                final_key = f"{key}_{bump}"
                bump += 1
            seen_keys.add(final_key)

            mapping = entry.get("mapping") or {}
            coord_mode = (entry.get("coord_mode") or "latlng").lower()
            sheet = entry.get("sheet")

            errors = validate_mapping(mapping, coord_mode)
            if errors:
                raise ValueError(
                    f"Mapping for '{saved_as}' is incomplete: " + "; ".join(errors)
                )

            canonical_name = f"{final_key}__canonical.csv"
            dest = uploads_dir / canonical_name
            try:
                materialize_dataset(
                    src_path=src,
                    dest_path=dest,
                    mapping=mapping,
                    coord_mode=coord_mode,
                    sheet=sheet,
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to apply mapping for '{saved_as}': {exc}"
                ) from exc

            rides_paths[final_key] = str(dest)
            original_filenames.append(entry.get("original_name") or saved_as)
    except ValueError as exc:
        return _render_error(
            job_id,
            "We couldn't apply your column mapping.",
            str(exc),
            http_status=400,
        )

    # Optional parking-hubs overlay.
    parking_path: str | None = None
    parking_label: str = ""
    parking_entry = payload.get("parking")
    if parking_entry:
        try:
            saved_as = (parking_entry.get("saved_as") or "").strip()
            if not saved_as:
                raise ValueError("Parking entry needs saved_as.")
            if "/" in saved_as or "\\" in saved_as or saved_as.startswith(".."):
                raise ValueError(f"Invalid parking file reference: {saved_as}")
            src = uploads_dir / saved_as
            if not src.exists():
                raise ValueError(
                    f"Parking file '{saved_as}' is no longer available. "
                    "Please re-upload."
                )
            mapping = parking_entry.get("mapping") or {}
            sheet = parking_entry.get("sheet")
            errors = validate_parking_mapping(mapping)
            if errors:
                raise ValueError(
                    "Parking mapping is incomplete: " + "; ".join(errors)
                )
            dest = uploads_dir / "PARKING__canonical.csv"
            try:
                materialize_parking_dataset(
                    src_path=src,
                    dest_path=dest,
                    mapping=mapping,
                    sheet=sheet,
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to apply parking mapping: {exc}"
                ) from exc
            parking_path = str(dest)
            parking_label = (parking_entry.get("label") or "").strip()
        except ValueError as exc:
            return _render_error(
                job_id,
                "We couldn't apply your parking-hub mapping.",
                str(exc),
                http_status=400,
            )

    log.info(
        "generate: job=%s operators=%s parking=%s workdir=%s",
        job_id,
        list(rides_paths.keys()),
        bool(parking_path),
        workdir,
    )

    download_name = _build_download_name(original_filenames)
    sync = request.args.get("sync") == "1" or request.form.get("sync") == "1"
    if sync:
        try:
            output_html = _run_notebook(
                rides_paths,
                workdir,
                parking_path=parking_path,
                parking_label=parking_label,
            )
            with open(output_html, "rb") as fh:
                blob = fh.read()
            return send_file(
                io.BytesIO(blob),
                mimetype="text/html",
                as_attachment=True,
                download_name=download_name,
            )
        except CellExecutionError as exc:
            log.exception("job=%s notebook execution failed", job_id)
            return _render_error(job_id, "The analysis notebook raised an error while processing your data.", _trim(str(exc)), http_status=500)
        except Exception as exc:
            log.exception("job=%s failed", job_id)
            return _render_error(job_id, "Unexpected server error while generating the report.", str(exc), http_status=500)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    _job_gc()
    _job_set(job_id, status="running", started=time.time(), download_name=download_name, workdir=str(workdir))
    worker = threading.Thread(
        target=_run_job,
        args=(job_id, rides_paths, workdir, parking_path, parking_label),
        daemon=True,
        name=f"standab-job-{job_id}",
    )
    worker.start()
    return (
        jsonify({"job_id": job_id, "status": "running", "status_url": f"/status/{job_id}", "result_url": f"/result/{job_id}"}),
        202,
    )


def _trim(text: str, keep: int = 2000) -> str:
    return text if len(text) <= 2 * keep else text[:keep] + "\n...[truncated]...\n" + text[-keep:]


def _run_job(job_id, rides_paths, workdir, parking_path, parking_label):
    """Run the notebook for a job in the background and record the outcome."""
    try:
        output_html = _run_notebook(
            rides_paths,
            workdir,
            parking_path=parking_path,
            parking_label=parking_label,
        )
        try:
            os.utime(workdir, None)          # keep the result for a full TTL from completion
        except OSError:
            pass
        _job_set(job_id, status="done", finished=time.time(), output=str(output_html))
        log.info("job=%s done: %s (%.1f MB)", job_id, output_html, Path(output_html).stat().st_size / 1e6)
    except CellExecutionError as exc:
        log.exception("job=%s notebook execution failed", job_id)
        _job_set(job_id, status="error", finished=time.time(), message="The analysis notebook raised an error while processing your data.", details=_trim(str(exc)))
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as exc:
        log.exception("job=%s failed", job_id)
        msg = "The notebook kernel died, most likely out of memory on this server. A smaller or shorter export, or the local runner, will work." if "Kernel died" in str(exc) or "DeadKernel" in type(exc).__name__ else "Unexpected server error while generating the report."
        _job_set(job_id, status="error", finished=time.time(), message=msg, details=str(exc))
        shutil.rmtree(workdir, ignore_errors=True)


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    jid = _sanitize_job_id(job_id)
    job = _job_get(jid) if jid else {}
    if not job:
        return jsonify({"status": "unknown", "message": "No such job, or it has expired."}), 404
    now = time.time()
    return jsonify({
        "job_id": jid,
        "status": job.get("status"),
        "elapsed_s": int((job.get("finished") or now) - job.get("started", now)),
        "message": job.get("message"),
        "details": job.get("details"),
        "download_name": job.get("download_name"),
    })


@app.route("/result/<job_id>", methods=["GET"])
def job_result(job_id: str):
    jid = _sanitize_job_id(job_id)
    job = _job_get(jid) if jid else {}
    if not job:
        return _render_error(jid, "No such job, or it has expired.", "Reports are kept for an hour after they are generated.", http_status=404)
    if job.get("status") != "done" or not job.get("output") or not Path(job["output"]).exists():
        return _render_error(jid, "The report is not ready.", f"Job status: {job.get('status')}.", http_status=409)
    return send_file(job["output"], mimetype="text/html", as_attachment=True, download_name=job.get("download_name") or "standab-analysis.html")


def _render_error(job_id: str | None, message: str, details: str, http_status: int):
    return (
        render_template(
            "error.html",
            job_id=job_id or "—",
            message=message,
            details=details,
        ),
        http_status,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
