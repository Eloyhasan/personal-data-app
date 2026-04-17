import io
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st

st.set_page_config(page_title="Personal Data App", layout="wide")

EXPECTED_COLUMNS = ["Date", "Weight", "Steps", "Mood", "Notes"]


def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL bulunamadı.")
    return psycopg2.connect(db_url)


def ensure_table_exists():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS health_data (
            id SERIAL PRIMARY KEY,
            date DATE,
            weight DOUBLE PRECISION,
            steps INTEGER,
            mood INTEGER,
            notes TEXT,
            UNIQUE (date, weight, steps, mood, notes)
        )
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    rename_map = {
        "date": "Date",
        "weight": "Weight",
        "steps": "Steps",
        "mood": "Mood",
        "notes": "Notes",
    }
    df = df.rename(columns=rename_map)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Eksik kolonlar var: {', '.join(missing)}. "
            f"Excel'de şu kolonlar olmalı: {', '.join(EXPECTED_COLUMNS)}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Steps"] = pd.to_numeric(df["Steps"], errors="coerce")
    df["Mood"] = pd.to_numeric(df["Mood"], errors="coerce")
    df["Notes"] = df["Notes"].fillna("").astype(str).str.strip()

    df = df.sort_values("Date", na_position="last").reset_index(drop=True)

    df["Weight_Change"] = df["Weight"] - df["Weight"].ffill().shift(1)
    df.loc[df["Weight"].isna(), "Weight_Change"] = np.nan

    return df


def insert_dataframe_to_db(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    conn = get_connection()
    cur = conn.cursor()
    inserted_count = 0

    for _, row in df.iterrows():
        row_date = row["Date"]
        row_weight = row["Weight"]
        row_steps = row["Steps"]
        row_mood = row["Mood"]
        row_notes = row["Notes"]

        row_date = None if pd.isna(row_date) else pd.to_datetime(row_date).date()
        row_weight = None if pd.isna(row_weight) else float(row_weight)
        row_steps = None if pd.isna(row_steps) else int(row_steps)
        row_mood = None if pd.isna(row_mood) else int(row_mood)
        row_notes = "" if pd.isna(row_notes) else str(row_notes).strip()

        cur.execute(
            """
            INSERT INTO health_data (date, weight, steps, mood, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date, weight, steps, mood, notes) DO NOTHING
            """,
            (row_date, row_weight, row_steps, row_mood, row_notes),
        )

        if cur.rowcount > 0:
            inserted_count += 1

    conn.commit()
    cur.close()
    conn.close()

    return inserted_count


def load_data_from_db() -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT date, weight, steps, mood, notes
        FROM health_data
        ORDER BY date, id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return prepare_dataframe(df)


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    display_df["Date"] = display_df["Date"].apply(
        lambda x: x.strftime("%d.%m.%Y") if pd.notna(x) else ""
    )
    display_df["Weight"] = display_df["Weight"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else ""
    )
    display_df["Steps"] = display_df["Steps"].apply(
        lambda x: f"{int(x)}" if pd.notna(x) else ""
    )
    display_df["Mood"] = display_df["Mood"].apply(
        lambda x: f"{int(x)}" if pd.notna(x) else ""
    )
    display_df["Weight_Change"] = display_df["Weight_Change"].apply(
        lambda x: f"{x:+.1f}" if pd.notna(x) else ""
    )

    return display_df


def style_weight_change(val):
    if val == "":
        return ""

    try:
        numeric_val = float(val.replace(",", "."))
    except Exception:
        return ""

    if numeric_val < 0:
        return "background-color: #9BE38C; color: black; font-weight: 700;"
    if numeric_val > 0:
        return "background-color: #FF8E8E; color: black; font-weight: 700;"
    return "background-color: #F2F2F2; color: black; font-weight: 700;"


def get_summary_metrics(df: pd.DataFrame):
    valid_weights = df["Weight"].dropna()
    valid_steps = df["Steps"].dropna()
    valid_mood = df["Mood"].dropna()

    last_weight = round(valid_weights.iloc[-1], 1) if not valid_weights.empty else None
    first_weight = round(valid_weights.iloc[0], 1) if not valid_weights.empty else None
    total_change = (
        round(valid_weights.iloc[-1] - valid_weights.iloc[0], 1)
        if len(valid_weights) >= 2
        else None
    )
    total_steps = int(valid_steps.sum()) if not valid_steps.empty else 0
    avg_mood = round(valid_mood.mean(), 1) if not valid_mood.empty else None

    return {
        "last_weight": last_weight,
        "first_weight": first_weight,
        "total_change": total_change,
        "total_steps": total_steps,
        "avg_mood": avg_mood,
    }


def get_rule_based_insights(df: pd.DataFrame):
    insights = []

    weight_df = df.dropna(subset=["Weight"]).copy()
    steps_df = df.dropna(subset=["Steps"]).copy()
    mood_df = df.dropna(subset=["Mood"]).copy()

    last_3_weights = weight_df.tail(3)
    if len(last_3_weights) >= 3:
        first_w = last_3_weights["Weight"].iloc[0]
        last_w = last_3_weights["Weight"].iloc[-1]

        if last_w < first_w:
            insights.append(("positive", "Momentum olumlu: Son 3 kilo kaydında düşüş trendi var."))
        elif last_w > first_w:
            insights.append(("warning", "Dikkat: Son 3 kilo kaydında yükseliş trendi var."))
        else:
            insights.append(("neutral", "Son 3 kilo kaydında yatay seyir var."))

    last_5_weights = weight_df.tail(5)
    if len(last_5_weights) >= 5:
        total_change_5 = last_5_weights["Weight"].iloc[-1] - last_5_weights["Weight"].iloc[0]
        if -0.2 <= total_change_5 <= 0.2:
            insights.append(("warning", "Plato riski: Son 5 kilo kaydında anlamlı değişim yok."))
        elif total_change_5 <= -0.5:
            insights.append(("positive", "Son 5 kilo kaydında belirgin düşüş var."))

    last_7_steps = steps_df.tail(7)
    if len(last_7_steps) >= 3:
        avg_steps = last_7_steps["Steps"].mean()
        if avg_steps >= 10000:
            insights.append(("positive", f"Aktivite güçlü: Son kayıtların ortalama adımı {avg_steps:,.0f}.".replace(",", ".")))
        elif avg_steps < 8000:
            insights.append(("warning", f"Aktivite düşük: Son kayıtların ortalama adımı {avg_steps:,.0f}.".replace(",", ".")))
        else:
            insights.append(("neutral", f"Aktivite orta seviyede: Ortalama adım {avg_steps:,.0f}.".replace(",", ".")))

    last_7_mood = mood_df.tail(7)
    if len(last_7_mood) >= 3:
        avg_mood = last_7_mood["Mood"].mean()
        if avg_mood >= 7:
            insights.append(("positive", f"Mood iyi seviyede: Ortalama {avg_mood:.1f}."))
        elif avg_mood < 6:
            insights.append(("warning", f"Mood düşük: Ortalama {avg_mood:.1f}. Sürdürülebilirlik riski olabilir."))
        else:
            insights.append(("neutral", f"Mood dengeli: Ortalama {avg_mood:.1f}."))

    if len(weight_df) >= 4:
        recent_changes = weight_df["Weight"].diff().tail(3)
        if (recent_changes > 0).sum() >= 2:
            insights.append(("warning", "Son 3 değişimin en az 2'si artış yönünde. Düzen gözden geçirilmeli."))
        elif (recent_changes < 0).sum() >= 2:
            insights.append(("positive", "Son 3 değişimin çoğu düşüş yönünde. Düzen iyi çalışıyor görünüyor."))

    if not insights:
        insights.append(("neutral", "Yeterli veri oluştuğunda burada akıllı yorumlar görünecek."))

    return insights


def render_insight_box(level: str, text: str):
    if level == "positive":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.info(text)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    export_df = df.copy()
    export_df["Date"] = export_df["Date"].dt.date

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Data")

    return output.getvalue()


st.title("Personal Data App")
st.caption("Excel yükle, veriyi DB'ye kaydet, dashboard'u ve akıllı yorumları uygulama oluştursun.")

try:
    ensure_table_exists()
except Exception as e:
    st.error(f"Veritabanı bağlantı hatası: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Excel dosyasını yükle", type=["xlsx", "xls"], accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Sheet seç", excel_file.sheet_names, index=0)

        raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        prepared_df = prepare_dataframe(raw_df)

        inserted_count = insert_dataframe_to_db(prepared_df)
        st.success(f"Upload tamamlandı. Yeni eklenen kayıt sayısı: {inserted_count}")

    except Exception as e:
        st.error(f"Excel işlenirken hata oluştu: {e}")
        st.stop()

with st.expander("Yeni kayıt ekle", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        new_date = st.date_input("Tarih", value=date.today())
        new_weight = st.number_input(
            "Kilo", min_value=0.0, max_value=300.0, value=90.0, step=0.1
        )

    with col2:
        new_steps = st.number_input(
            "Adım", min_value=0, max_value=100000, value=10000, step=500
        )
        new_mood = st.number_input(
            "Mood", min_value=0, max_value=100, value=10, step=1
        )

    with col3:
        new_notes = st.text_input("Not", value="")

    if st.button("Kaydı ekle"):
        new_row = pd.DataFrame(
            [
                {
                    "Date": pd.to_datetime(new_date),
                    "Weight": float(new_weight),
                    "Steps": int(new_steps),
                    "Mood": int(new_mood),
                    "Notes": new_notes.strip(),
                }
            ]
        )

        try:
            inserted_count = insert_dataframe_to_db(new_row)
            if inserted_count > 0:
                st.success("Yeni kayıt DB'ye eklendi.")
            else:
                st.warning("Bu kayıt zaten vardı, tekrar eklenmedi.")
            st.rerun()
        except Exception as e:
            st.error(f"Kayıt eklenirken hata oluştu: {e}")

try:
    df = load_data_from_db()
except Exception as e:
    st.error(f"DB'den veri okunurken hata oluştu: {e}")
    st.stop()

if df.empty:
    st.info("Henüz veri yok. Excel yükleyebilir veya yeni kayıt ekleyebilirsin.")
    st.stop()

# 1) EN ÜSTTE KILO TRENDI
st.markdown("### Kilo Trendi")

weight_chart_df = df.dropna(subset=["Date", "Weight"]).copy()
if not weight_chart_df.empty:
    fig_weight = px.line(
        weight_chart_df,
        x="Date",
        y="Weight",
        markers=True,
        title="",
    )
    st.plotly_chart(fig_weight, use_container_width=True)

# 2) HEMEN ALTINDA TABLO
st.markdown("### Kayıtlar")

display_df = format_display_df(df)
styled_df = display_df.style.map(style_weight_change, subset=["Weight_Change"])

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# 3) SONRA ÖZET METRIKLER
metrics = get_summary_metrics(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Son Kilo", f"{metrics['last_weight']:.1f}" if metrics["last_weight"] is not None else "-")
m2.metric("Başlangıç Kilo", f"{metrics['first_weight']:.1f}" if metrics["first_weight"] is not None else "-")
m3.metric(
    "Toplam Kilo Değişimi",
    f"{metrics['total_change']:+.1f}" if metrics["total_change"] is not None else "-",
)
m4.metric("Ortalama Mood", f"{metrics['avg_mood']}" if metrics["avg_mood"] is not None else "-")

# 4) AKILLI YORUMLAR
st.markdown("### Akıllı Yorumlar")

insights = get_rule_based_insights(df)
for level, text in insights:
    render_insight_box(level, text)

# 5) EN ALTA DIGER GRAFIKLER
st.markdown("### Diğer Grafikler")

g1, g2 = st.columns(2)

with g1:
    steps_chart_df = df.dropna(subset=["Date", "Steps"]).copy()
    if not steps_chart_df.empty:
        fig_steps = px.bar(
            steps_chart_df,
            x="Date",
            y="Steps",
            title="Adım Trendi",
        )
        st.plotly_chart(fig_steps, use_container_width=True)

with g2:
    mood_chart_df = df.dropna(subset=["Date", "Mood"]).copy()
    if not mood_chart_df.empty:
        fig_mood = px.line(
            mood_chart_df,
            x="Date",
            y="Mood",
            markers=True,
            title="Mood Trendi",
        )
        st.plotly_chart(fig_mood, use_container_width=True)

st.markdown("### Dışa aktar")

col_a, col_b = st.columns(2)

with col_a:
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV indir",
        data=csv_data,
        file_name="personal_data_processed.csv",
        mime="text/csv",
    )

with col_b:
    excel_bytes = to_excel_bytes(df)
    st.download_button(
        "Excel indir",
        data=excel_bytes,
        file_name="personal_data_processed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
