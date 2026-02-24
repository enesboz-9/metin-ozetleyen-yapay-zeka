import streamlit as st
from model_helper import ai_engine # Doğrudan dosyadan çağırıyoruz

st.set_page_config(page_title="AI Özetleyici", page_icon="📝")

st.title("📝 Akıllı Metin Özetleyici")

# Modeli belleğe al (Uygulama her yenilendiğinde tekrar yüklenmesin diye)
@st.cache_resource
def load_ai():
    ai_engine.load_model()
    return ai_engine

model = load_ai()

text_input = st.text_area("Özetlenecek Metin", placeholder="Buraya yapıştırın...", height=250)
max_len = st.select_slider("Özet Uzunluğu", options=[50, 100, 150, 200], value=100)

if st.button("Özeti Oluştur"):
    if text_input and len(text_input) >= 50:
        with st.spinner('Yapay zeka çalışıyor...'):
            # API yerine doğrudan fonksiyonu çağırıyoruz
            result = model.summarize(text_input, max_len)
            
            st.success(f"Dili Algılandı: {result['detected_language'].upper()}")
            st.subheader("🤖 Özet")
            st.write(result['summary'])
    else:
        st.warning("En az 50 karakter giriniz.")
