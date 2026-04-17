import os
import streamlit as st
import psycopg2

st.title("Personal Data App")

db_url = os.getenv("DATABASE_URL")

if not db_url:
    st.error("DATABASE_URL bulunamadı")
    st.stop()

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # tablo oluştur
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)
    conn.commit()

    # input
    note = st.text_input("Not gir")

    if st.button("Kaydet"):
        if note.strip():
            cur.execute("INSERT INTO notes (text) VALUES (%s)", (note,))
            conn.commit()
            st.success("Kaydedildi")
        else:
            st.warning("Boş not olmaz")

    # listele
    cur.execute("SELECT * FROM notes ORDER BY id DESC")
    rows = cur.fetchall()

    st.subheader("Kayıtlar")
    for row in rows:
        st.write(f"{row[0]} - {row[1]}")

    cur.close()
    conn.close()

except Exception as e:
    st.error(f"Hata: {e}")
