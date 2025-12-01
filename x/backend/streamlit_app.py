# frontend.py
import streamlit as st
import requests
import pandas as pd

# ------------------ CONFIG ------------------
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Twitter NLQ CSV Scraper", layout="wide")

st.title("Twitter NLQ CSV Scraper")
st.write("Enter a natural language query to fetch tweets and download as CSV.")

# ------------------ INPUTS ------------------
with st.form("nl_search_form"):
    nl_query = st.text_input("Natural Language Query", placeholder="e.g., '10 tweets from @elonmusk about SpaceX in October 2023'")
    fetch_max = st.number_input("Fetch Max (hard capped at 10)", min_value=1, max_value=10, value=10)
    default_count = st.number_input("Default Count (hard capped at 10)", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Search Tweets")

# ------------------ SUBMIT ------------------
if submitted:
    if not nl_query.strip():
        st.error("Please enter a natural language query.")
    else:
        payload = {
            "nl_query": nl_query.strip(),
            "fetch_max": int(fetch_max),
            "default_count": int(default_count)
        }

        try:
            with st.spinner("Fetching tweets..."):
                resp = requests.post(f"{BACKEND_URL}/nl_search_csv", json=payload, timeout=30)
            if resp.status_code != 200:
                error_json = resp.json()
                st.error(f"API Error: {error_json.get('error', 'Unknown error')}")
            else:
                data = resp.json()
                tweets = data.get("tweets", [])
                if not tweets:
                    st.warning("No tweets found for this query.")
                else:
                    st.success(f"Found {len(tweets)} tweets for @{data.get('username')}")

                    # Show results as table
                    df = pd.DataFrame(tweets)
                    df_display = df[["created_at", "text", "like_count", "retweet_count", "reply_count", "quote_count", "score"]]
                    st.dataframe(df_display)

                    # Show CSV download link
                    csv_url = data.get("csv_url")
                    if csv_url:
                        full_csv_url = f"{BACKEND_URL}{csv_url}" if not csv_url.startswith("http") else csv_url
                        st.markdown(f"[Download CSV]({full_csv_url})")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend: {e}")
