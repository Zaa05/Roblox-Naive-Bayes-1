import streamlit as st
import joblib
import re
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# NLTK DOWNLOAD (CLOUD SAFE)
# ===============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
model = joblib.load("bernoulli_nb.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===============================
# PREPROCESSING (SAMA DENGAN COLAB)
# ===============================
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ===============================
# LABEL SENTIMEN (3 KELAS)
# ===============================
label_map = {
    0: "Negatif 😠",
    1: "Netral 😐",
    2: "Positif 😊"
}

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Analisis Sentimen", page_icon="💬")

st.title("💬 Analisis Sentimen Teks Bahasa Indonesia")
st.write("Model: **Bernoulli Naive Bayes + TF-IDF**")

text_input = st.text_area("Masukkan teks:")

if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        clean_text = preprocess(text_input)
        vector = tfidf.transform([clean_text])
        prediction = model.predict(vector)[0]

        st.success(f"Sentimen: **{label_map[prediction]}**")
