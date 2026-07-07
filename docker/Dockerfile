FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash netron

COPY --chown=netron:netron source source

USER netron

EXPOSE 8080

CMD ["python", "-c", "import source; source.main()", "--host", "0.0.0.0", "--port", "8080"]