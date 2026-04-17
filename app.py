import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st

st.set_page_config(page_title="Smart Body Tracker", layout="wide")

EXPECTED_COLUMNS = ["Date", "Weight", "Steps", "Mood", "Notes"]
TARGET_WEIGHT = 88  # hedef kilo

# ---------------- DB ----------------

def get_connection():
    db_url = os.getenv("DATABASE_URL")
    return psycopg2.connect(db_url)

def ensure_table_exists():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS health_data (
        id SERIAL PRIMARY KEY,
        date DATE,
        weight FLOAT,
        steps INTEGER,
        mood INTEGER,
        notes TEXT,
        UNIQUE (date, weight, steps, mood, notes)
    )
    """)

    conn.commit()
    cur.close()
    conn.close()

# ---------------- DATA ----------------

def prepare_dataframe(df):
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "date": "Date",
        "weight": "Weight",
        "steps": "Steps",
        "mood": "Mood",
        "notes": "Notes",
    }
    df = df.rename(columns=rename_map)

    df = df[EXPECTED_COLUMNS]

    df["Date"] = pd.to_datetime(df["Date"])
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Steps"] = pd.to_numeric(df["Steps"], errors="coerce")
    df["Mood"] = pd.to_numeric(df["Mood"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    df["Weight_Change"] = df["Weight"] - df["Weight"].shift(1)

    return df

def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT date, weight, steps, mood, notes FROM health_data ORDER BY date", conn)
    conn.close()
    return prepare_dataframe(df)

def insert_df(df):
    conn = get_connection()
    cur = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
        INSERT INTO health_data (date, weight, steps, mood, notes)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """, (
            row["Date"].date(),
            row["Weight"],
            row["Steps"],
            row["Mood"],
            row["Notes"]
        ))

    conn.commit()
    cur.close()
    conn.close()

# ---------------- PREDICTION ----------------

def calculate_prediction(df):
    weight_df = df.dropna(subset=["Weight"]).tail(10)

    if len(weight_df) < 3:
        return None, None, None

    weight_df = weight_df.copy()
    weight_df["t"] = np.arange(len(weight_df))

    x = weight_df["t"].values
    y = weight_df["Weight"].values

    # lineer regresyon
    slope, intercept = np.polyfit(x, y, 1)

    # 7 gün sonrası tahmin
    future_days = 7
    future_t = np.arange(len(weight_df), len(weight_df) + future_days)
    future_weights = slope * future_t + intercept

    last_weight = y[-1]

    # hedefe kaç gün
    if slope >= 0:
        days_to_target = None
    else:
        days_to_target = int((TARGET_WEIGHT - last_weight) / slope)

    return future_weights, slope, days_to_target

# ---------------- UI ----------------

st.title("Smart Body Tracker")
st.caption("Track. Analyze. Improve.")

ensure_table_exists()

uploaded_file = st.file_uploader("Excel yükle", type=["xlsx"])

if uploaded_file:
    df_upload = pd.read_excel(uploaded_file)
    df_upload = prepare_dataframe(df_upload)
    insert_df(df_upload)
    st.success("Data yüklendi")

df = load_data()

if df.empty:
    st.stop()

# ---------------- GRAPH (TOP) ----------------

st.markdown("### Kilo Trendi + Tahmin")

future_weights, slope, days_to_target = calculate_prediction(df)

fig = go.Figure()

# gerçek veri
fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Weight"],
    mode='lines+markers',
    name="Gerçek"
))

# tahmin
if future_weights is not None:
    last_date = df["Date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_weights))]

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_weights,
        mode='lines',
        name="Tahmin",
        line=dict(dash="dash")
    ))

st.plotly_chart(fig, use_container_width=True)

# ---------------- TABLE ----------------

st.markdown("### Kayıtlar")
st.dataframe(df, use_container_width=True)

# ---------------- METRICS ----------------

last_weight = df["Weight"].dropna().iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Son Kilo", f"{last_weight:.1f}")

if days_to_target:
    col2.metric("Hedefe Gün", f"{days_to_target}")
else:
    col2.metric("Hedefe Gün", "Trend negatif değil")

col3.metric("Trend", f"{slope:.3f} kg/gün")

# ---------------- RULE INSIGHTS ----------------

st.markdown("### Akıllı Yorumlar")

if slope < 0:
    st.success("Kilo düşüş trendinde 👍")
elif slope > 0:
    st.warning("Kilo artış trendi var ⚠️")
else:
    st.info("Trend sabit")

if days_to_target:
    st.info(f"Bu tempoyla hedefe yaklaşık {days_to_target} gün var")

# ---------------- EXTRA ----------------

st.markdown("### Diğer Grafikler")

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(df["Steps"])

with col2:
    st.line_chart(df["Mood"])
