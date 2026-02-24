import streamlit as st
from model_helper import ai_engine

st.set_page_config(page_title="AI Ozetleyici", page_icon="📝")

st.title("📝 Akıllı Metin Özetleyici")

# Cache yerine her butona basıldığında (eğer yüklü değilse) yüklemesini sağlayalım
text_input = st.text_area("Metni buraya girin...", height=200)
max_len = st.slider("Ozet Uzunlugu", 50, 200, 100)

if st.button("Ozetle"):
    if text_input and len(text_input) > 50:
        with st.spinner('Islem yapiliyor...'):
            try:
                # Doğrudan model_helper içindeki summarize'ı çağırıyoruz
                res = ai_engine.summarize(text_input, max_len)
                st.success(f"Dil: {res['detected_language'].upper()}")
                st.write(res['summary'])
            except Exception as e:
                st.error(f"Hata detayi: {e}")
    else:
        st.warning("Lutfen yeterli uzunlukta metin girin.")
