FROM python:3.12-alpine

RUN pip install --no-cache-dir netron

EXPOSE 8080

CMD ["netron", "--host", "0.0.0.0", "--port", "8080"]
