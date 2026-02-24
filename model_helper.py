# model_helper.py dosyasındaki model_name kısmını değiştiriyoruz
class SummarizerModel:
    def __init__(self):
        # mT5 yerine daha hafif olan t5-small tabanlı bir model deneyelim
        self.model_name = "google/t5-v1_1-small" 
        self.device = -1 # Streamlit Cloud'da GPU olmadığı için -1 (CPU) kalsın
        self.pipe = None
