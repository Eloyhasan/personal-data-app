import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Personal Data App", layout="wide")

EXPECTED_COLUMNS = ["Date", "Weight", "Steps", "Mood", "Notes"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

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
    df["Notes"] = df["Notes"].fillna("").astype(str)

    df = df.sort_values("Date", na_position="last").reset_index(drop=True)

    # Weight boş değilse önceki dolu kiloya göre fark hesapla
    df["Weight_Change"] = df["Weight"] - df["Weight"].ffill().shift(1)
    df.loc[df["Weight"].isna(), "Weight_Change"] = np.nan

    return df


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


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    export_df = df.copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Data")

    return output.getvalue()


st.title("Personal Data App")
st.caption("Excel yükle, hesaplamaları uygulama yapsın.")

uploaded_file = st.file_uploader(
    "Excel dosyasını yükle", type=["xlsx", "xls"], accept_multiple_files=False
)

if "working_df" not in st.session_state:
    st.session_state.working_df = None

if uploaded_file is not None:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Sheet seç", excel_file.sheet_names, index=0)

        raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        prepared_df = prepare_dataframe(raw_df)
        st.session_state.working_df = prepared_df.copy()

    except Exception as e:
        st.error(f"Excel okunurken hata oluştu: {e}")
        st.stop()

if st.session_state.working_df is None:
    st.info("Başlamak için uygun kolonlara sahip Excel dosyanı yükle.")
    st.markdown(
        """
Gerekli kolonlar:
- `Date`
- `Weight`
- `Steps`
- `Mood`
- `Notes`
"""
    )
    st.stop()

df = st.session_state.working_df.copy()

with st.expander("Yeni kayıt ekle", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        new_date = st.date_input("Tarih", value=date.today())
        new_weight = st.number_input("Kilo", min_value=0.0, max_value=300.0, value=90.0, step=0.1)
    with col2:
        new_steps = st.number_input("Adım", min_value=0, max_value=100000, value=10000, step=500)
        new_mood = st.number_input("Mood", min_value=0, max_value=100, value=10, step=1)
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
        combined = pd.concat([df[EXPECTED_COLUMNS], new_row], ignore_index=True)
        combined = prepare_dataframe(combined)
        st.session_state.working_df = combined
        st.success("Yeni kayıt eklendi.")
        st.rerun()

metrics = get_summary_metrics(df)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Son Kilo", f"{metrics['last_weight']:.1f}" if metrics["last_weight"] is not None else "-")
m2.metric("Başlangıç Kilo", f"{metrics['first_weight']:.1f}" if metrics["first_weight"] is not None else "-")
m3.metric(
    "Toplam Kilo Değişimi",
    f"{metrics['total_change']:+.1f}" if metrics["total_change"] is not None else "-",
)
m4.metric("Ortalama Mood", f"{metrics['avg_mood']}" if metrics["avg_mood"] is not None else "-")

st.markdown("### Kayıtlar")

display_df = format_display_df(df)
styled_df = display_df.style.map(style_weight_change, subset=["Weight_Change"])

st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown("### Grafikler")

g1, g2 = st.columns(2)

with g1:
    weight_chart_df = df.dropna(subset=["Date", "Weight"]).copy()
    if not weight_chart_df.empty:
        fig_weight = px.line(
            weight_chart_df,
            x="Date",
            y="Weight",
            markers=True,
            title="Kilo Trendi",
        )
        st.plotly_chart(fig_weight, use_container_width=True)

with g2:
    steps_chart_df = df.dropna(subset=["Date", "Steps"]).copy()
    if not steps_chart_df.empty:
        fig_steps = px.bar(
            steps_chart_df,
            x="Date",
            y="Steps",
            title="Adım Trendi",
        )
        st.plotly_chart(fig_steps, use_container_width=True)

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
