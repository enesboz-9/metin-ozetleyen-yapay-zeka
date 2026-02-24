from transformers import pipeline
from langdetect import detect, DetectorFactory
import torch

DetectorFactory.seed = 0

class SummarizerModel:
    def __init__(self):
        # En stabil ve hafif model
        self.model_name = "t5-small" 
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            # CPU üzerinde çalışması için zorluyoruz
            self.pipe = pipeline(
                "summarization",
                model=self.model_name,
                device=-1
            )
        return self.pipe

    def detect_language(self, text: str):
        try:
            return detect(text)
        except:
            return "unknown"

    def summarize(self, text: str, max_len: int = 100):
        if self.pipe is None:
            self.load_model()
            
        lang = self.detect_language(text)
        
        # T5-small için metnin önüne 'summarize: ' eklemek performansı artırır
        input_text = f"summarize: {text}"
        
        result = self.pipe(
            input_text,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )
        
        return {
            "summary": result[0]['summary_text'],
            "detected_language": lang
        }

ai_engine = SummarizerModel()
