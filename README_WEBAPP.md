# Standab Web App

Tiny Flask app that lets users upload ride CSVs and download an interactive
analysis HTML, without ever exposing the underlying Python code.

## How it works

1. User uploads one or more CSVs in the browser (one per operator).
2. `app.py` saves them to a per-request temp directory.
3. It loads `Standab Data Analysis Tool.ipynb`, patches the cell that
   hardcodes `RIDES_CSV_PATHS` so it points at the uploaded files, and
   executes the whole notebook with `nbclient`.
4. The notebook writes `maps/output.html` inside the temp dir; the app
   streams that file back as an attachment and deletes the temp dir.

Because the notebook runs server-side, the client only ever receives the
generated self-contained HTML.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5001
```

The app listens on `$PORT` if set, otherwise `5001`. Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `STANDAB_EXEC_TIMEOUT` | `900` | Per-cell timeout (seconds) for notebook execution. |

There is no hard upload size limit enforced by the app — the practical
ceiling is whatever your reverse proxy / platform allows (e.g. Railway's
edge proxy) and how much memory the notebook worker has.

## Deploying to Railway

The repo is set up for Railway's Dockerfile builder — `railway.json` tells
Railway to build `Dockerfile` and run a `/health` check on each deploy.
Gunicorn binds to `0.0.0.0:${PORT}`, which Railway injects at runtime.

### First-time setup

1. Install the Railway CLI and log in:
   ```bash
   npm i -g @railway/cli
   railway login
   ```
2. From the repo root, link (or create) a project:
   ```bash
   railway init     # pick "Empty Project" and a name like standab-web
   railway link     # if the project already exists
   ```
3. Set env vars (optional — defaults work):
   ```bash
   railway variables --set STANDAB_EXEC_TIMEOUT=900
   ```
4. Deploy:
   ```bash
   railway up
   ```
5. Generate a public domain for the service:
   ```bash
   railway domain
   ```

### Recommended service settings

- **Instance size:** at least 2 GB RAM. The notebook loads pandas + scipy +
  folium and builds in-memory H3 hex grids; a 512 MB instance will OOM on
  anything beyond a toy dataset.
- **HTTP timeout:** bump the service's request timeout to ~600 s (the
  maximum Railway allows). Notebook runs can take a couple of minutes on
  large CSVs.
- **Sleep / scale-to-zero:** fine to leave on; cold-start adds ~3–5 s.

### Redeploying after changes

Pushing a new commit to the linked Git branch (if you connect the repo in
the Railway dashboard) auto-deploys. Otherwise re-run `railway up` from the
repo root. Infrastructure stays the same regardless of whether you changed
the notebook, `app.py`, or the UI.

## What gets hidden from users

- No Python source is ever sent to the browser.
- `operator_dataset_toggle.py` and the notebook stay on the server.
- Uploaded CSVs land in a per-job `/tmp/standab-<id>-*` directory that is
  deleted in the `finally` block (`shutil.rmtree(..., ignore_errors=True)`).
- The downloaded HTML is pure rendered output — Folium map, embedded data
  as base64/zlib-compressed JSON, and client-side JS — no server code.
