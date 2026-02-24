from transformers import pipeline
from langdetect import detect, DetectorFactory
import torch

# Dil algılamanın her seferinde aynı sonucu vermesi için (kararlılık)
DetectorFactory.seed = 0

class SummarizerModel:
    def __init__(self):
        self.model_name = "csebuetnlp/mT5_multilingual_XLSum"
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            print(f"--- Model ve Dil Algılayıcı Yükleniyor ---")
            self.pipe = pipeline(
                "summarization",
                model=self.model_name,
                device=self.device
            )
            print("--- Sistem Hazır ---")
        return self.pipe

    def detect_language(self, text: str):
        """Metnin dilini algılar (tr, en, de vb.)"""
        try:
            return detect(text)
        except:
            return "unknown"

    def summarize(self, text: str, max_len: int = 100):
        if self.pipe is None:
            self.load_model()
            
        # Dil tespiti yap
        lang = self.detect_language(text)
        
        result = self.pipe(
            text,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )
        
        return {
            "summary": result[0]['summary_text'],
            "detected_language": lang
        }

ai_engine = SummarizerModel()
