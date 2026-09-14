from __future__ import annotations

import os
import re
import sqlite3
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DB_PATH = Path("retailiq.db")
CSV_PATH = Path("data/superstore.csv")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SCHEMA = """
CREATE TABLE sales (
  order_id TEXT,
  order_date DATE,
  ship_date DATE,
  ship_mode TEXT,
  customer_name TEXT,
  segment TEXT,
  city TEXT,
  state TEXT,
  region TEXT,
  product_name TEXT,
  category TEXT,
  sub_category TEXT,
  sales REAL,
  quantity INTEGER,
  discount REAL,
  profit REAL
)
"""

COLUMN_MAP = {
    "Order ID": "order_id",
    "Order Date": "order_date",
    "Ship Date": "ship_date",
    "Ship Mode": "ship_mode",
    "Customer Name": "customer_name",
    "Segment": "segment",
    "City": "city",
    "State": "state",
    "Region": "region",
    "Product Name": "product_name",
    "Category": "category",
    "Sub-Category": "sub_category",
    "Sub Category": "sub_category",
    "Sales": "sales",
    "Quantity": "quantity",
    "Discount": "discount",
    "Profit": "profit",
}
REQUIRED_COLUMNS = list(dict.fromkeys(COLUMN_MAP.values()))
BLOCKED_SQL = {"alter", "attach", "create", "delete", "detach", "drop", "insert", "pragma", "replace", "update"}

SAMPLE_QUESTIONS = [
    "Which region has highest profit margin?",
    "Show monthly sales trend for 2017",
    "Top 10 cities by revenue",
    "Which category has most discounting?",
    "Compare segment performance by profit",
    "Which shipping mode is most used?",
    "States with negative profit",
    "Best month for sales historically",
]


st.set_page_config(page_title="RetailIQ", page_icon="RIQ", layout="wide")


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing Superstore columns: {missing}")

    df = df[REQUIRED_COLUMNS]
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.date
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce").dt.date
    for column in ["sales", "quantity", "discount", "profit"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    df["quantity"] = df["quantity"].astype(int)
    return df


@st.cache_resource
def load_database(csv_bytes: bytes | None) -> sqlite3.Connection:
    source = BytesIO(csv_bytes) if csv_bytes else CSV_PATH
    if not csv_bytes and not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {CSV_PATH}")

    df = normalize_data(pd.read_csv(source, encoding="latin1"))
    conn = sqlite3.connect(":memory:" if csv_bytes else DB_PATH, check_same_thread=False)
    df.to_sql("sales", conn, if_exists="replace", index=False)
    conn.commit()
    return conn


@st.cache_data
def get_metrics(csv_bytes: bytes | None) -> dict[str, object]:
    conn = load_database(csv_bytes)
    totals = pd.read_sql(
        """
        SELECT
          SUM(sales) AS revenue,
          COUNT(DISTINCT order_id) AS orders,
          SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0) AS aov
        FROM sales
        """,
        conn,
    ).iloc[0]
    top_region = pd.read_sql(
        "SELECT region, SUM(sales) AS revenue FROM sales GROUP BY region ORDER BY revenue DESC LIMIT 1",
        conn,
    )
    return {
        "revenue": float(totals["revenue"] or 0),
        "orders": int(totals["orders"] or 0),
        "aov": float(totals["aov"] or 0),
        "top_region": top_region.iloc[0]["region"] if not top_region.empty else "N/A",
    }


def validate_sql(sql: str) -> str:
    sql = sql.strip().strip("`").replace("sql\n", "").rstrip(";")
    lowered = sql.lower()
    tokens = set(re.findall(r"[a-z_]+", lowered))
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if blocked := sorted(BLOCKED_SQL.intersection(tokens)):
        raise ValueError(f"Blocked unsafe SQL keyword: {', '.join(blocked)}")
    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")
    return f"{sql};"


def groq_client() -> OpenAI:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def question_to_sql(question: str) -> str:
    response = groq_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a SQL expert. Convert the user question to a valid SQLite query for the sales table. "
                    "Return ONLY the SQL query, nothing else. Never use DROP, DELETE, or UPDATE."
                ),
            },
            {"role": "user", "content": f"Schema:\n{SCHEMA}\n\nQuestion: {question}"},
        ],
    )
    return validate_sql(response.choices[0].message.content or "")


def generate_insight(question: str, df: pd.DataFrame) -> str:
    response = groq_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a business analyst. Given this data result, write a 2-3 sentence business insight "
                    "that directly answers the question. Be specific with numbers."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nResult CSV:\n{df.head(30).to_csv(index=False)}\nRows: {len(df)}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def fallback_sql(question: str) -> str:
    q = question.lower()
    if "profit margin" in q and "region" in q:
        sql = "SELECT region, ROUND(SUM(profit) * 100.0 / NULLIF(SUM(sales), 0), 2) AS profit_margin_percent FROM sales GROUP BY region ORDER BY profit_margin_percent DESC"
    elif "monthly sales trend" in q:
        sql = "SELECT strftime('%Y-%m', order_date) AS month, SUM(sales) AS revenue FROM sales WHERE strftime('%Y', order_date) = '2017' GROUP BY month ORDER BY month"
    elif "top 10 cities" in q or ("cities" in q and "revenue" in q):
        sql = "SELECT city, SUM(sales) AS revenue FROM sales GROUP BY city ORDER BY revenue DESC LIMIT 10"
    elif "discount" in q and "category" in q:
        sql = "SELECT category, ROUND(AVG(discount) * 100, 2) AS avg_discount_percent FROM sales GROUP BY category ORDER BY avg_discount_percent DESC"
    elif "segment" in q and "profit" in q:
        sql = "SELECT segment, SUM(profit) AS profit, SUM(sales) AS revenue FROM sales GROUP BY segment ORDER BY profit DESC"
    elif "shipping mode" in q or "ship mode" in q:
        sql = "SELECT ship_mode, COUNT(*) AS orders FROM sales GROUP BY ship_mode ORDER BY orders DESC"
    elif "negative profit" in q and "state" in q:
        sql = "SELECT state, SUM(profit) AS profit FROM sales GROUP BY state HAVING SUM(profit) < 0 ORDER BY profit ASC"
    elif "best month" in q and "sales" in q:
        sql = "SELECT strftime('%m', order_date) AS month_number, SUM(sales) AS revenue FROM sales GROUP BY month_number ORDER BY revenue DESC LIMIT 1"
    else:
        sql = "SELECT category, SUM(sales) AS revenue FROM sales GROUP BY category ORDER BY revenue DESC"
    return validate_sql(sql)


def fallback_insight(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows matched this question."
    numeric = list(df.select_dtypes(include="number").columns)
    if not numeric:
        return f"The query returned {len(df):,} rows. Review the result table for details."
    metric = numeric[-1]
    row = df.sort_values(metric, ascending=False).iloc[0]
    label = str(row[[column for column in df.columns if column != metric][0]])
    return f"{label} leads with {metric} of {row[metric]:,.2f}. The result includes {len(df):,} returned rows from the active sales dataset."


def run_query(question: str, conn: sqlite3.Connection) -> dict[str, object]:
    used_fallback = False
    try:
        sql = question_to_sql(question)
    except Exception:
        used_fallback = True
        sql = fallback_sql(question)

    df = pd.read_sql_query(validate_sql(sql), conn)
    chart = make_chart(df.copy())

    try:
        insight = fallback_insight(df) if used_fallback else generate_insight(question, df)
    except Exception:
        used_fallback = True
        insight = fallback_insight(df)

    return {"sql": sql, "df": df, "chart": chart, "insight": insight, "used_fallback": used_fallback}


def make_chart(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return None
    numeric = list(df.select_dtypes(include="number").columns)
    date_col = next((column for column in df.columns if "date" in column.lower() or "month" in column.lower()), None)
    if date_col and numeric:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return px.line(df.sort_values(date_col), x=date_col, y=numeric[0], markers=True)
    if len(df.columns) == 2 and numeric:
        metric = numeric[0]
        label = next(column for column in df.columns if column != metric)
        if label.lower() in {"category", "sub_category"} and len(df) < 6:
            return px.pie(df, names=label, values=metric)
        return px.bar(df, x=label, y=metric)
    return None


def main() -> None:
    st.title("RetailIQ")
    st.caption("Natural language BI for Superstore sales using Groq, SQLite, pandas, and Plotly.")

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("question", SAMPLE_QUESTIONS[0])
    st.session_state.setdefault("result", None)

    uploaded = st.sidebar.file_uploader("Upload Superstore CSV", type=["csv"])
    csv_bytes = uploaded.getvalue() if uploaded else None
    dataset_name = uploaded.name if uploaded else str(CSV_PATH)

    if st.session_state.get("dataset_name") != dataset_name:
        st.session_state.dataset_name = dataset_name
        st.session_state.result = None

    st.sidebar.subheader("Query History")
    for item in st.session_state.history[-10:][::-1]:
        if st.sidebar.button(item, use_container_width=True):
            st.session_state.question = item

    try:
        conn = load_database(csv_bytes)
        metrics = get_metrics(csv_bytes)
    except Exception as exc:
        st.error(f"Could not load dataset. {exc}")
        st.stop()

    st.caption(f"Active dataset: {dataset_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${metrics['revenue']:,.0f}")
    c2.metric("Total Orders", f"{metrics['orders']:,}")
    c3.metric("Avg Order Value", f"${metrics['aov']:,.0f}")
    c4.metric("Top Region", metrics["top_region"])

    chip_cols = st.columns(4)
    for index, sample in enumerate(SAMPLE_QUESTIONS):
        if chip_cols[index % 4].button(sample, use_container_width=True):
            st.session_state.question = sample

    with st.form("ask"):
        question = st.text_input("Ask a business question", key="question")
        submitted = st.form_submit_button("Ask RetailIQ", type="primary")

    if submitted and question.strip():
        with st.spinner("Generating SQL, running SQLite, and preparing insight..."):
            st.session_state.result = run_query(question.strip(), conn)
            st.session_state.history.append(question.strip())
            st.session_state.history = st.session_state.history[-10:]

    if st.session_state.result:
        result = st.session_state.result
        tab1, tab2, tab3 = st.tabs(["Insight", "Chart", "SQL"])
        with tab1:
            if result["used_fallback"]:
                st.warning("Groq was unavailable, so RetailIQ used a local fallback.")
            st.write(result["insight"])
            st.download_button(
                "Export result CSV",
                result["df"].to_csv(index=False).encode("utf-8"),
                "retailiq_result.csv",
                "text/csv",
            )
        with tab2:
            if result["chart"] is None:
                st.dataframe(result["df"], use_container_width=True)
            else:
                st.plotly_chart(result["chart"], use_container_width=True)
                st.dataframe(result["df"], use_container_width=True)
        with tab3:
            st.code(result["sql"], language="sql")


if __name__ == "__main__":
    main()
