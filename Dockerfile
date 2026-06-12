FROM nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/home/innov-xc/Projects/lhy/data-agents/.venv \
    PATH=/home/innov-xc/Projects/lhy/data-agents/.venv/bin:/home/innov-xc/miniconda3/bin:$PATH

WORKDIR /home/innov-xc/Projects/lhy/data-agents

COPY .venv /home/innov-xc/Projects/lhy/data-agents/.venv
COPY configs /home/innov-xc/Projects/lhy/data-agents/configs
COPY src /home/innov-xc/Projects/lhy/data-agents/src
COPY docker /home/innov-xc/Projects/lhy/data-agents/docker
COPY pyproject.toml README.md README.zh.md /home/innov-xc/Projects/lhy/data-agents/
COPY docker-runtime/miniconda3 /home/innov-xc/miniconda3

RUN chmod +x /home/innov-xc/Projects/lhy/data-agents/docker/entrypoint.sh

ENTRYPOINT ["/home/innov-xc/Projects/lhy/data-agents/docker/entrypoint.sh"]
