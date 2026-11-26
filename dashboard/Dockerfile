# Stakeholder + monitoring dashboard (same app as Streamlit Community Cloud)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt
COPY dashboard/ dashboard/
COPY src/__init__.py src/config.py src/
COPY src/model/ src/model/
COPY models/ models/
RUN useradd --create-home dash && chown -R dash /app
USER dash
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
