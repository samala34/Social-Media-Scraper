
import streamlit as st
import pandas as pd
import re
import sqlite3
from datetime import datetime

# ---------------------------
# Default hashtag categories
# ---------------------------

WOMENSDAY_TAGS = [
    "#womensday",
    "#internationalwomensday",
    "#iwd",
    "#iwd2024",
    "#iwd2025",
    "#womenempowerment",
    "#womenpower",
    "#womensupportingwomen",
    "#girlpower",
    "#womeninbusiness",
    "#womeninspiringwomen",
    "#strongwomen",
    "#womenempoweringwomen",
    "#feminism",
    "#womenshistorymonth",
    "#who_run_the_world_girls",
    "#womenrights",
    "#womensrights",
    "#sheinspiresme",
    "#herstory",
    "#thefutureisfemale",
    "#womenofworld",
    "#supportwomen",
    "#empowerher",
    "#feminist",
    "#sheleads",
    "#womeninleadership",
    "#womeninstem",
    "#genderquality",
    "#heforshe",
    "#eachforequal",
    "#choose_to_challenge",
    "#womenintech",
    "#womenleaders",
    "#womenunite",
    "#empoweringgirls",
    "#girlsupportgirls",
    "#womenmakinghistory",
    "#femalefounders",
    "#bossbabe",
    "#breakingthebias",
    "#womensdaycelebration",
    "#happywomensday",
    "#womensday2024",
    "#womensday2025",
    "#womensdayspecial",
    "#womenforchange",
    "#shepersists",
    "#motivationforwomen",
    "#womenrise",
    "#futureisfemale"
]

AI_TAGS = [
    "#ai",
    "#artificialintelligence",
    "#machinelearning",
    "#deeplearning",
    "#neuralnetworks",
    "#datascience",
    "#bigdata",
    "#automation",
    "#aitech",
    "#aitools",
    "#aiethics",
    "#nlp",
    "#naturallanguageprocessing",
    "#computervision",
    "#digitaltransformation",
    "#aicommunity",
    "#aifuture",
    "#aiinnovation",
    "#airesearch",
    "#ainews",
    "#aidevelopment",
    "#agi",
    "#generativeai",
    "#gans",
    "#promptengineering",
    "#promptdesign",
    "#aiart",
    "#aidesign",
    "#robotics",
    "#robot",
    "#smarttech",
    "#techinnovation",
    "#aiapplications",
    "#aiinbusiness",
    "#aiinhealthcare",
    "#aiinfinance",
    "#supervisedlearning",
    "#unsupervisedlearning",
    "#reinforcementlearning",
    "#predictiveanalytics",
    "#algorithm",
    "#analytics",
    "#aistartup",
    "#aiindustry",
    "#aiworld",
    "#aiexpert",
    "#techforgood",
    "#aiproducts",
    "#aitrends",
    "#mlengineer",
    "#aisolutions",
    "#aidriven",
    "#aimodel",
    "#businessintelligence",
    "#aiconference",
    "#aichatbot",
    "#openai",
    "#chatgpt"
]

DB_PATH = "history.db"


# ---------------------------
# DB helpers
# ---------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            filename TEXT,
            filter_mode TEXT,
            year_filter TEXT,
            month_filter TEXT,
            tags_used TEXT,
            rows_before INTEGER,
            rows_after INTEGER
        )
        """
    )
    conn.commit()
    return conn


def log_run(conn, filename, filter_mode, year_filter, month_filter,
            tags_used, rows_before, rows_after):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO runs (
            created_at, filename, filter_mode, year_filter, month_filter,
            tags_used, rows_before, rows_after
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            filename,
            filter_mode,
            year_filter,
            month_filter,
            ", ".join(tags_used) if tags_used else "",
            rows_before,
            rows_after,
        ),
    )
    conn.commit()


# ---------------------------
# Filtering logic
# ---------------------------

def prepare_dataframe(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Drop known unused columns if present
    for col in ["firstComment", "queryTag"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Normalise caption column
    if "caption" not in df.columns:
        st.error("No 'caption' column found in the uploaded CSV.")
        return None

    df["caption"] = df["caption"].astype(str).str.lower()

    # Handle timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def apply_filters(df, tags, year_filter, month_filter):
    # Hashtag filter
    tags = [t.strip().lower() for t in tags if t.strip()]
    if not tags:
        st.warning("No hashtags provided. Returning original data.")
        filtered = df.copy()
    else:
        pattern = re.compile("|".join([re.escape(t) for t in tags]))
        filtered = df[df["caption"].apply(lambda x: bool(pattern.search(str(x))))]

    # Date filters (if timestamp exists)
    if "timestamp" in filtered.columns and filtered["timestamp"].notna().any():
        if year_filter:
            filtered = filtered[filtered["timestamp"].dt.year.isin(year_filter)]
        if month_filter:
            filtered = filtered[filtered["timestamp"].dt.month.isin(month_filter)]

    return filtered


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="Post Filter Tool", layout="wide")
    st.title("📊 Social Media Post Filtering Tool")

    st.markdown(
        "Upload any CSV containing at least a **caption** column (and optionally **timestamp**) "
        "and filter posts by hashtags and date."
    )

    conn = init_db()

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to get started.")
        return

    df = prepare_dataframe(uploaded_file)
    if df is None:
        return

    st.subheader("📁 Dataset Overview")
    st.write(f"Rows: **{len(df)}**, Columns: **{len(df.columns)}**")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("🎯 Filtering Options")

    # Filter mode
    filter_mode = st.radio(
        "Hashtag mode",
        ["Women's Day", "AI", "Custom / Mixed"],
        help="Choose a preset category or define your own hashtags."
    )

    tags = []
    if filter_mode == "Women's Day":
        st.caption("Using built-in Women's Day hashtag set (you can still add more below).")
        tags = WOMENSDAY_TAGS.copy()
    elif filter_mode == "AI":
        st.caption("Using built-in AI hashtag set (you can still add more below).")
        tags = AI_TAGS.copy()

    custom_tags_text = st.text_area(
        "Extra/custom hashtags (comma or space separated)",
        value="",
        placeholder="#womensday, #ai, #example"
    )

    if custom_tags_text.strip():
        # Split by comma or whitespace
        extra = re.split(r"[,\s]+", custom_tags_text.strip())
        tags.extend(extra)

    # Remove duplicates while preserving order
    seen = set()
    uniq_tags = []
    for t in tags:
        t = t.strip()
        if not t:
            continue
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq_tags.append(t)

    st.markdown("**Active hashtag filters:**")
    if uniq_tags:
        st.code(", ".join(uniq_tags))
    else:
        st.warning("Currently no hashtags selected. If you continue, no hashtag filtering will be applied.")

    # Date filters - ask user while using the application (dynamic)
    st.markdown("### 🗓 Date Filters (optional)")

    use_date_filter = st.checkbox("Enable date filtering (uses 'timestamp' column if available)")

    year_filter = []
    month_filter = []

    if use_date_filter and "timestamp" in df.columns and df["timestamp"].notna().any():
        available_years = sorted(y for y in df["timestamp"].dt.year.dropna().unique())
        year_filter = st.multiselect("Filter by year", available_years, default=available_years)

        available_months = sorted(m for m in df["timestamp"].dt.month.dropna().unique())
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        month_labels = [f"{m} - {month_names.get(m, '')}" for m in available_months]
        month_map = dict(zip(month_labels, available_months))

        selected_month_labels = st.multiselect(
            "Filter by month",
            month_labels,
            default=month_labels
        )
        month_filter = [month_map[label] for label in selected_month_labels]
    elif use_date_filter:
        st.error("No valid 'timestamp' column found or it contains only invalid dates. Date filter will be ignored.")

    st.markdown("### 📤 Output Format")
    export_csv = st.checkbox("Export as CSV", value=True)
    export_json = st.checkbox("Export as JSON", value=True)

    if not export_csv and not export_json:
        st.warning("Select at least one export format (CSV or JSON).")

    if st.button("🔍 Run Filter"):
        rows_before = len(df)
        filtered_df = apply_filters(df, uniq_tags, year_filter, month_filter)
        rows_after = len(filtered_df)

        st.subheader("✅ Filter Result")
        st.write(f"Rows before: **{rows_before}**, after filtering: **{rows_after}**")

        if rows_after == 0:
            st.warning("No rows matched the selected filters.")
        else:
            st.dataframe(filtered_df.head(100))

            if export_csv:
                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name="filtered_posts.csv",
                    mime="text/csv"
                )

            if export_json:
                json_str = filtered_df.to_json(orient="records", force_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name="filtered_posts.json",
                    mime="application/json"
                )

        # Log run in DB
        filename = getattr(uploaded_file, "name", "uploaded.csv")
        year_str = ",".join(str(y) for y in year_filter) if year_filter else ""
        month_str = ",".join(str(m) for m in month_filter) if month_filter else ""
        log_run(conn, filename, filter_mode, year_str, month_str, uniq_tags, rows_before, rows_after)

    st.markdown("---")
    st.subheader("🕒 Previous Runs (local history)")

    cur = conn.cursor()
    cur.execute("SELECT created_at, filename, filter_mode, year_filter, month_filter, rows_before, rows_after FROM runs ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    if rows:
        history_df = pd.DataFrame(
            rows,
            columns=["created_at", "filename", "filter_mode", "year_filter", "month_filter", "rows_before", "rows_after"]
        )
        st.dataframe(history_df)
    else:
        st.info("No history yet. Run a filter to populate this table.")


if __name__ == "__main__":
    main()
