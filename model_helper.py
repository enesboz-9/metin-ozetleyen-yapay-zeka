from transformers import pipeline
from langdetect import detect, DetectorFactory
import torch

DetectorFactory.seed = 0

class SummarizerModel:
    def __init__(self):
        # facebook/bart-large-cnn çok güçlüdür ama ağırdır. 
        # Biz en hafif ve standart olan 'sshleifer/distilbart-cnn-12-6' deneyelim.
        # Bu model özetlemede t5-small'dan daha kararlıdır.
        self.model_name = "sshleifer/distilbart-cnn-12-6" 
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            try:
                print(f"Model yukleniyor: {self.model_name}")
                self.pipe = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1 # CPU zorlaması
                )
                print("Model basariyla yuklendi!")
            except Exception as e:
                # Hatayı siyah ekrana (logs) yazdırıyoruz
                print(f"KRITIK MODEL HATASI: {str(e)}")
                self.pipe = None
        return self.pipe

    def detect_language(self, text: str):
        try:
            return detect(text)
        except:
            return "unknown"

    def summarize(self, text: str, max_len: int = 100):
        model = self.load_model()
        
        if model is None:
            return {"summary": "Model dosyasi sunucuda olusturulamadi (RAM yetersiz olabilir).", "detected_language": "Bilinmiyor"}

        lang = self.detect_language(text)
        
        try:
            result = model(
                text,
                max_length=max_len,
                min_length=30,
                do_sample=False
            )
            return {
                "summary": result[0]['summary_text'],
                "detected_language": lang
            }
        except Exception as e:
            print(f"Ozetleme aninda hata: {str(e)}")
            return {"summary": f"Ozetleme hatasi: {str(e)}", "detected_language": lang}

ai_engine = SummarizerModel()
