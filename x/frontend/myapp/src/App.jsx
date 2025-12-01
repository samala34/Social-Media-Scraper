// frontend/src/App.jsx
import React, { useState } from "react";

export default function App() {
  const [nlq, setNlq] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [tweets, setTweets] = useState([]);
  const [meta, setMeta] = useState(null);

  async function handleSearch(e) {
    e && e.preventDefault();
    setErr("");
    setLoading(true);
    setTweets([]);
    setMeta(null);
    try {
      const resp = await fetch("http://localhost:8000/nl_search_csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nl_query: nlq, fetch_max: 500, default_count: 10 })
      });
      const data = await resp.json();
      if (!resp.ok) {
        setErr(data.error || JSON.stringify(data));
      } else {
        setTweets(data.tweets || []);
        setMeta({
          username: data.username,
          requested_query: data.requested_query,
          used_topic: data.used_topic,
          count_returned: data.count_returned,
          csv_filename: data.csv_filename || null,
          csv_url: data.csv_url || null,
          debug_parse: data.debug_parse || null
        });
      }
    } catch (e) {
      setErr(e.message || "Network error");
    } finally {
      setLoading(false);
    }
  }

  async function handleDownloadCsv() {
    if (!meta || !meta.csv_filename) {
      alert("No CSV available");
      return;
    }
    const url = `http://localhost:8000/download_csv/${meta.csv_filename}`;
    try {
      const resp = await fetch(url);
      if (!resp.ok) {
        const txt = await resp.text();
        alert("Download failed: " + resp.status + " " + txt);
        return;
      }
      const blob = await resp.blob();
      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = meta.csv_filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(href);
    } catch (e) {
      alert("Download error: " + e.message);
    }
  }

  return (
    <div style={{ maxWidth: 980, margin: "28px auto", fontFamily: 'Inter, -apple-system, Roboto, "Segoe UI", Helvetica, Arial, sans-serif' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>Twitter Scraper</h1>
        <div style={{ fontSize: 13, color: '#666' }}>Backend: http://localhost:8000</div>
      </header>

      <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input
          value={nlq}
          onChange={(e) => setNlq(e.target.value)}
          placeholder="e.g. Give me 10 tweets of Microsoft regarding AI on 22-6-2025"
          style={{
            flex: 1,
            padding: '10px 12px',
            borderRadius: 8,
            border: '1px solid #ddd',
            fontSize: 15
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: '10px 16px',
            borderRadius: 8,
            border: 'none',
            background: '#0b5fff',
            color: 'white'
          }}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {err && <div style={{ color: "crimson", marginBottom: 12 }}>{err}</div>}

      {meta && (
        <div style={{ marginBottom: 12 }}>
          <strong>Query:</strong> {meta.requested_query} <br />
          <strong>Username:</strong> @{meta.username} • <strong>Topic used:</strong> {meta.used_topic} • <strong>Returned:</strong> {meta.count_returned}
          <div style={{ marginTop: 8 }}>
            {meta.csv_filename && (
              <button onClick={handleDownloadCsv} style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #ddd', background: 'white' }}>
                Download CSV ({meta.csv_filename})
              </button>
            )}
          </div>
        </div>
      )}

      <div style={{ border: '1px solid #eee', borderRadius: 10, overflow: 'hidden' }}>
        {tweets.length === 0 && !loading && <div style={{ padding: 18, color: '#666' }}>No results — run a query above.</div>}
        {tweets.map((t) => (
          <article key={t.id} style={{ padding: 14, borderBottom: '1px solid #f6f6f6', background: 'white' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, color: '#555' }}>
                  {t.created_at ? new Date(t.created_at).toLocaleString() : ""} • Score: {typeof t.score === 'number' ? t.score.toFixed(3) : t.score}
                </div>
                <div style={{ marginTop: 8, fontSize: 15 }}>{t.text}</div>
                <div style={{ marginTop: 10, fontSize: 13, color: '#444' }}>
                  <strong>Likes:</strong> {t.like_count} &nbsp; • &nbsp;
                  <strong>Retweets:</strong> {t.retweet_count} &nbsp; • &nbsp;
                  <strong>Replies:</strong> {t.reply_count} &nbsp; • &nbsp;
                  <strong>Quotes:</strong> {t.quote_count}
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <a href={`https://twitter.com/${meta?.username || ""}/status/${t.id}`} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>
                  <button style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #ddd', background: 'white' }}>Open</button>
                </a>
                <button onClick={() => navigator.clipboard?.writeText(t.text)} style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #ddd', background: 'white' }}>Copy</button>
              </div>
            </div>
          </article>
        ))}
      </div>

      {meta?.debug_parse && (
        <details style={{ marginTop: 12 }}>
          <summary>Debug parse</summary>
          <pre style={{ whiteSpace: "pre-wrap", padding: 12 }}>{JSON.stringify(meta.debug_parse, null, 2)}</pre>
        </details>
      )}

      <footer style={{ marginTop: 18, fontSize: 12, color: '#777' }}>
        Tip: backend must be running at <code>http://localhost:8000</code> and CSVs served from that process.
      </footer>
    </div>
  );
}
