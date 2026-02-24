from transformers import pipeline
from langdetect import detect, DetectorFactory

# Dil tespiti için kararlılık
DetectorFactory.seed = 0

class SummarizerModel:
    def __init__(self):
        # En hafif ve uyumlu model
        self.model_name = "t5-small" 
        self.pipe = None

    def load_model(self):
        """Modeli sadece ihtiyaç anında ve CPU üzerinde yükler."""
        if self.pipe is None:
            try:
                self.pipe = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1 # CPU zorlaması
                )
            except Exception as e:
                print(f"Model yukleme hatasi: {e}")
        return self.pipe

    def summarize(self, text: str, max_len: int = 100):
        # Modeli yükle (yüklenmemişse)
        model = self.load_model()
        
        if model is None:
            return {"summary": "Model yuklenemedi.", "detected_language": "Bilinmiyor"}

        # Dil tespiti
        try:
            lang = detect(text)
        except:
            lang = "unknown"
        
        # Özetleme işlemi
        # T5 modeli 'summarize: ' ön ekiyle daha iyi çalışır
        input_text = f"summarize: {text}"
        
        result = model(
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
