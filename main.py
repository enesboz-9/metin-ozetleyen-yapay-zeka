from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# 1. FastAPI uygulamasını başlat
app = FastAPI(title="AI Metin Özetleyici")

# 2. Özetleme modelini yükle (İlk çalıştırmada modeli indirir)
# 'facebook/bart-large-cnn' oldukça başarılı bir modeldir
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 3. Veri modelini tanımla
class TextRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

@app.get("/")
def home():
    return {"mesaj": "Metin Özetleme API'sine Hoş Geldiniz! /docs adresine giderek test edebilirsiniz."}

@app.post("/summarize")
async def summarize_text(request: TextRequest):
    try:
        # Metin çok kısaysa hata ver
        if len(request.text) < 50:
            raise HTTPException(status_code=400, detail="Özetlemek için metin çok kısa.")

        # Modeli çalıştır
        summary = summarizer(
            request.text, 
            max_length=request.max_length, 
            min_length=request.min_length, 
            do_sample=False
        )
        
        return {
            "original_length": len(request.text),
            "summary_length": len(summary[0]['summary_text']),
            "summary": summary[0]['summary_text']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))