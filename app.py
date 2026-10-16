from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fake_news_detector import load_model, predict_text

ROOT = Path(__file__).parent
artifact = load_model(ROOT / "models" / "fake_news_model.joblib")

app = FastAPI()


class Article(BaseModel):
    text: str


@app.post("/predict")
def predict(article: Article):
    # Returns {"label": ..., "confidence": ..., "probabilities": {...}}
    return predict_text(artifact["pipeline"], article.text)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/index.js")
def script():
    return FileResponse(ROOT / "index.js")


@app.get("/style.css")
def styles():
    return FileResponse(ROOT / "style.css")
