FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    IPYTHONDIR=/tmp/standab_ipython \
    JUPYTER_CONFIG_DIR=/tmp/standab_jupyter \
    PORT=8080

# Minimal system deps. The heavy scientific Python wheels already ship
# manylinux binaries, so we only need libgomp (used by scipy) and curl
# for container healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make pip resilient to slow/flaky PyPI reads on the build network.
# Railway's builder intermittently drops download streams mid-file
# (ReadTimeout / IncompleteRead). We combat this three ways:
#   * PIP_DEFAULT_TIMEOUT/PIP_RETRIES  - longer per-read timeout + connection retries
#   * --resume-retries                 - resume a download that broke mid-stream
#   * an outer retry loop              - re-run the whole install on any failure
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN set -e; \
    for i in 1 2 3 4 5; do \
      echo "=== pip install attempt $i/5 ==="; \
      if pip install --retries 10 --timeout 120 --resume-retries 5 \
           -r requirements.txt "gunicorn>=21.2"; then \
        echo "pip install succeeded"; \
        break; \
      fi; \
      if [ "$i" = "5" ]; then echo "pip install failed after 5 attempts"; exit 1; fi; \
      echo "attempt $i failed; retrying in 10s..."; \
      sleep 10; \
    done

# Copy the rest of the project. .dockerignore excludes heavy/irrelevant files.
COPY . .

# Pre-create the Jupyter/IPython runtime dirs so the first request doesn't
# have to wait for them to be created under /tmp.
RUN mkdir -p /tmp/standab_ipython /tmp/standab_jupyter

EXPOSE 8080

# Gunicorn config:
#   * 1 sync worker  - notebook execution is CPU-bound; more workers would
#     just multiply memory usage.
#   * --timeout 900  - notebook runs can take a couple of minutes on large
#     datasets.
#   * --graceful-timeout 30 - give the worker 30s to drain in-flight
#     requests during deploys.
#
# Shell form so ${PORT} is expanded at container start - Railway injects its
# own PORT env var which overrides the default 8080 above.
CMD gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 \
    --threads 2 \
    --timeout 900 \
    --graceful-timeout 30 \
    --keep-alive 10 \
    --access-logfile - \
    --error-logfile - \
    app:app
