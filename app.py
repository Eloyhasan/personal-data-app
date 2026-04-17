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
TARGET_WEIGHT = 88.0
FORECAST_DAYS = 7


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


def remove_weight_outliers(weight_df: pd.DataFrame) -> pd.DataFrame:
    """
    Median Absolute Deviation ile uç değer temizliği.
    Veri azsa olduğu gibi döner.
    """
    clean_df = weight_df.copy()
    if len(clean_df) < 5:
        return clean_df

    median_weight = clean_df["Weight"].median()
    abs_dev = np.abs(clean_df["Weight"] - median_weight)
    mad = np.median(abs_dev)

    if mad == 0 or np.isnan(mad):
        return clean_df

    modified_z = 0.6745 * (clean_df["Weight"] - median_weight) / mad
    clean_df = clean_df[np.abs(modified_z) <= 3.5].copy()

    if len(clean_df) < 3:
        return weight_df.copy()

    return clean_df


def calculate_prediction(df: pd.DataFrame):
    """
    Tahmin için:
    - son 10 dolu kilo kaydı
    - outlier temizleme
    - 3 noktalı smoothing
    - lineer trend
    - güven aralığı
    """
    weight_df = df.dropna(subset=["Weight"])[["Date", "Weight"]].tail(10).copy()

    if len(weight_df) < 3:
        return None

    filtered_df = remove_weight_outliers(weight_df)

    if len(filtered_df) < 3:
        filtered_df = weight_df.copy()

    filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)
    filtered_df["Weight_Smooth"] = filtered_df["Weight"].rolling(window=3, min_periods=1).mean()

    model_df = filtered_df.dropna(subset=["Weight_Smooth"]).copy()
    if len(model_df) < 3:
        return None

    model_df["t"] = np.arange(len(model_df))
    x = model_df["t"].values.astype(float)
    y = model_df["Weight_Smooth"].values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted

    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    if np.isnan(residual_std):
        residual_std = 0.0

    # Tahmin çizgisi son gerçek noktadan devam etsin
    future_t = np.arange(len(model_df) - 1, len(model_df) + FORECAST_DAYS)
    future_pred = slope * future_t + intercept

    # Güven bandı hafif genişleyerek artsın
    horizon_index = np.arange(len(future_t))
    ci_multiplier = np.sqrt(horizon_index + 1)
    ci_width = 1.96 * residual_std * ci_multiplier

    forecast_upper = future_pred + ci_width
    forecast_lower = future_pred - ci_width

    last_smoothed_weight = float(y[-1])

    if slope < 0:
        raw_days = (TARGET_WEIGHT - last_smoothed_weight) / slope
        days_to_target = int(np.ceil(raw_days)) if raw_days > 0 else 0
    else:
        days_to_target = None

    forecast_start_date = model_df["Date"].iloc[-1]
    future_dates = [forecast_start_date + timedelta(days=i) for i in range(len(future_t))]

    predicted_weight_7d = float(future_pred[-1])

    return {
        "slope": float(slope),
        "days_to_target": days_to_target,
        "future_dates": future_dates,
        "future_pred": future_pred,
        "forecast_upper": forecast_upper,
        "forecast_lower": forecast_lower,
        "predicted_weight_7d": predicted_weight_7d,
        "filtered_df": filtered_df,
        "model_df": model_df,
        "residual_std": residual_std,
    }


def build_prediction_chart(df: pd.DataFrame, prediction_result: dict | None):
    fig = go.Figure()

    actual_df = df.dropna(subset=["Date", "Weight"]).copy()

    # Gerçek veri
    fig.add_trace(
        go.Scatter(
            x=actual_df["Date"],
            y=actual_df["Weight"],
            mode="lines+markers",
            name="Gerçek",
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="Tarih: %{x|%d.%m.%Y}<br>Kilo: %{y:.1f}<extra></extra>",
        )
    )

    # Hedef çizgisi
    if not actual_df.empty and prediction_result is not None:
        x_end = prediction_result["future_dates"][-1]
    elif not actual_df.empty:
        x_end = actual_df["Date"].max() + timedelta(days=FORECAST_DAYS)
    else:
        x_end = date.today() + timedelta(days=FORECAST_DAYS)

    if not actual_df.empty:
        x_start = actual_df["Date"].min()
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[TARGET_WEIGHT, TARGET_WEIGHT],
                mode="lines",
                name=f"Hedef ({TARGET_WEIGHT:.1f})",
                line=dict(width=2, dash="dot"),
                hovertemplate="Hedef kilo: %{y:.1f}<extra></extra>",
            )
        )

    # Tahmin + güven aralığı
    if prediction_result is not None:
        future_dates = prediction_result["future_dates"]
        future_pred = prediction_result["future_pred"]
        forecast_upper = prediction_result["forecast_upper"]
        forecast_lower = prediction_result["forecast_lower"]

        # Güven bandı
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="Güven Aralığı",
                hoverinfo="skip",
                opacity=0.18,
            )
        )

        # Tahmin çizgisi
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_pred,
                mode="lines",
                name="Tahmin",
                line=dict(width=3, dash="dash"),
                hovertemplate="Tahmin tarihi: %{x|%d.%m.%Y}<br>Tahmini kilo: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis_title="",
        yaxis_title="Kilo",
    )

    fig.update_yaxes(tickformat=".1f")

    return fig


st.title("Smart Body Tracker")
st.caption("Track. Analyze. Improve.")

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

# 1) EN ÜSTTE PROFESYONEL TAHMİN GRAFİĞİ
st.markdown("### Kilo Trendi + Tahmin")

prediction_result = calculate_prediction(df)
prediction_chart = build_prediction_chart(df, prediction_result)
st.plotly_chart(prediction_chart, use_container_width=True)

if prediction_result is not None:
    st.caption(
        "Tahmin, son kilo kayıtlarının temizlenmiş ve yumuşatılmış trendine göre hesaplanır. "
        "Kesikli çizgi projeksiyonu, gölgeli alan yaklaşık güven aralığını gösterir."
    )
else:
    st.caption("Tahmin için en az 3 adet dolu kilo kaydı gerekir.")

# 2) TABLO
st.markdown("### Kayıtlar")

display_df = format_display_df(df)
styled_df = display_df.style.map(style_weight_change, subset=["Weight_Change"])
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# 3) ÖZET METRİKLER
metrics = get_summary_metrics(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Son Kilo", f"{metrics['last_weight']:.1f}" if metrics["last_weight"] is not None else "-")
m2.metric("Başlangıç Kilo", f"{metrics['first_weight']:.1f}" if metrics["first_weight"] is not None else "-")
m3.metric(
    "Toplam Kilo Değişimi",
    f"{metrics['total_change']:+.1f}" if metrics["total_change"] is not None else "-",
)

if prediction_result is not None:
    m4.metric("7 Gün Tahmini", f"{prediction_result['predicted_weight_7d']:.1f}")
else:
    m4.metric("7 Gün Tahmini", "-")

# 4) PREDICTION ÖZETİ
st.markdown("### Tahmin Özeti")

p1, p2, p3 = st.columns(3)

if prediction_result is not None:
    slope = prediction_result["slope"]
    days_to_target = prediction_result["days_to_target"]
    residual_std = prediction_result["residual_std"]

    if slope < 0:
        trend_text = f"{slope:.3f} kg/gün"
    elif slope > 0:
        trend_text = f"+{slope:.3f} kg/gün"
    else:
        trend_text = "0.000 kg/gün"

    p1.metric("Günlük Trend", trend_text)

    if days_to_target is not None:
        p2.metric("Hedefe Tahmini Gün", str(days_to_target))
    else:
        p2.metric("Hedefe Tahmini Gün", "Belirsiz")

    p3.metric("Model Oynaklığı", f"{residual_std:.2f}")
else:
    p1.metric("Günlük Trend", "-")
    p2.metric("Hedefe Tahmini Gün", "-")
    p3.metric("Model Oynaklığı", "-")

# 5) AKILLI YORUMLAR
st.markdown("### Akıllı Yorumlar")

insights = get_rule_based_insights(df)
for level, text in insights:
    render_insight_box(level, text)

if prediction_result is not None:
    slope = prediction_result["slope"]
    days_to_target = prediction_result["days_to_target"]

    if slope < 0:
        st.success("Tahmin motoru olumlu sinyal veriyor: kısa vadeli trend aşağı yönlü.")
    elif slope > 0:
        st.warning("Tahmin motoru dikkat uyarısı veriyor: kısa vadeli trend yukarı yönlü.")

    if days_to_target is not None:
        st.info(f"Mevcut tempoyla {TARGET_WEIGHT:.1f} kg hedefine yaklaşık {days_to_target} gün içinde ulaşabilirsin.")
    elif slope >= 0:
        st.warning("Mevcut tahmin eğilimi hedef kiloya yaklaşmayı göstermiyor. Aktivite ve düzen yeniden gözden geçirilmeli.")

# 6) DİĞER GRAFİKLER
st.markdown("### Diğer Grafikler")

g1, g2 = st.columns(2)

with g1:
    steps_chart_df = df.dropna(subset=["Date", "Steps"]).copy()
    if not steps_chart_df.empty:
        fig_steps = go.Figure()
        fig_steps.add_trace(
            go.Bar(
                x=steps_chart_df["Date"],
                y=steps_chart_df["Steps"],
                name="Adım",
                hovertemplate="Tarih: %{x|%d.%m.%Y}<br>Adım: %{y:.0f}<extra></extra>",
            )
        )
        fig_steps.update_layout(
            title="Adım Trendi",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="",
            yaxis_title="Adım",
        )
        st.plotly_chart(fig_steps, use_container_width=True)

with g2:
    mood_chart_df = df.dropna(subset=["Date", "Mood"]).copy()
    if not mood_chart_df.empty:
        fig_mood = go.Figure()
        fig_mood.add_trace(
            go.Scatter(
                x=mood_chart_df["Date"],
                y=mood_chart_df["Mood"],
                mode="lines+markers",
                name="Mood",
                hovertemplate="Tarih: %{x|%d.%m.%Y}<br>Mood: %{y:.0f}<extra></extra>",
            )
        )
        fig_mood.update_layout(
            title="Mood Trendi",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="",
            yaxis_title="Mood",
        )
        st.plotly_chart(fig_mood, use_container_width=True)

# 7) DIŞA AKTAR
st.markdown("### Dışa aktar")

col_a, col_b = st.columns(2)

with col_a:
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV indir",
        data=csv_data,
        file_name="smart_body_tracker_data.csv",
        mime="text/csv",
    )

with col_b:
    excel_bytes = to_excel_bytes(df)
    st.download_button(
        "Excel indir",
        data=excel_bytes,
        file_name="smart_body_tracker_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
