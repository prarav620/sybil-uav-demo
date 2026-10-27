FROM python:3.10-slim
WORKDIR /srv
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
# only what the demo needs: the feature code, the server, the page, the model and samples
COPY src/__init__.py src/__init__.py
COPY src/classifiers/__init__.py src/classifiers/__init__.py
COPY src/classifiers/baseline_rf.py src/classifiers/baseline_rf.py
COPY app/__init__.py app/server.py app/index.html app/model.joblib app/samples.json app/results.json app/
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
