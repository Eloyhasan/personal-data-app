import streamlit as st
import pandas as pd

st.title("Kişisel Veri Takip")

# veri giriş alanı
date = st.date_input("Tarih")
weight = st.number_input("Kilo", step=0.1)
steps = st.number_input("Adım")
mood = st.slider("Mood (1-10)", 1, 10)
notes = st.text_input("Not")

if st.button("Kaydet"):
    data = {
        "Date": date,
        "Weight": weight,
        "Steps": steps,
        "Mood": mood,
        "Notes": notes
    }

    df = pd.DataFrame([data])

    try:
        existing = pd.read_csv("data.csv")
        df = pd.concat([existing, df])
    except:
        pass

    df.to_csv("data.csv", index=False)

    st.success("Kayıt eklendi")

# veri göster
try:
    df = pd.read_csv("data.csv")
    st.dataframe(df)
except:
    st.info("Henüz veri yok")
