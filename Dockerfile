# Imagem só da API de predição: grupo principal do pyproject.toml, sem dados brutos nem notebooks.
#   docker build -t jorge-ml .
#   docker run --rm -p 8000:8000 jorge-ml
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY api.py predicao.py clima.py config.py ./
COPY resultados/modelos_cultura ./resultados/modelos_cultura

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
