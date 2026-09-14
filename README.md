# RetailIQ

RetailIQ is a compact Streamlit BI app. It turns plain-English questions into SQL
with Groq, runs the SQL on a local SQLite sales table, and returns an insight,
chart, SQL preview, and CSV export.

## Simple Structure

```text
RetailIQ/
|-- app.py
|-- data/
|   `-- superstore.csv
|-- retailiq.db
|-- requirements.txt
|-- .env
`-- README.md
```

`app.py` contains the database loading, SQL generation, SQL safety check, query
execution, charting, and Streamlit UI. A separate validator file is not required
for this MVP, but the safety check is still kept inside `app.py` because LLM SQL
must be blocked from running destructive statements.

## Setup

```powershell
cd "C:\Users\deepi\data project"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

`.env` should contain:

```text
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Dataset

The app loads `data/superstore.csv` by default. You can also upload a Superstore
CSV from the sidebar; the uploaded file replaces the default dataset for that
Streamlit session.
