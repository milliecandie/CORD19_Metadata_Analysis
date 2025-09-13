"""
CORD-19 Metadata Analysis - Full Assignment (Single Script)

Author: Millicent Nabututu Makokha
Date: 2025

This script:
- Downloads (or uses local) CORD-19 `metadata.csv`
- Loads a safe number of rows (adjustable)
- Cleans and prepares the data (dates, missing values, wordcounts)
- Performs basic analysis:
    * Publications by year
    * Top publishing journals
    * Frequent title words
- Generates visualizations saved in ./outputs/
- Writes a brief text report ./brief_report.txt
- Includes an optional Streamlit app to display the findings:
    Run with: streamlit run CORD19_full_assignment.py

Notes:
- The script is resilient: if `wordcloud` or `streamlit` are not installed it will skip those features and continue.
- Default NROWS is small for low-RAM devices. Increase on PC (set NROWS = None to read whole file).
- Required (core) packages: pandas, matplotlib, requests
- Optional packages: wordcloud, streamlit
"""

import os
import sys
import re
import requests
from datetime import datetime
from collections import Counter

# Matplotlib backend safe for headless environments (set before pyplot import)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Core data library
import pandas as pd

# Optional niceties (import if available)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# -----------------------
# Configuration
# -----------------------
AUTHOR_NAME = "Millicent Nabututu Makokha"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default safe small read for phones. Set to None on PC to load entire file.
NROWS = 2000

# Primary fallback metadata URL (if you want a different release, change this)
# Note: large file; script reads only first NROWS rows by default
METADATA_URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02/metadata.csv"
LOCAL_FILENAME = "metadata.csv"

# -----------------------
# Helper functions
# -----------------------
def download_metadata(url=METADATA_URL, dest=LOCAL_FILENAME, force=False):
    """Download metadata.csv if not present (or if force=True)."""
    if os.path.exists(dest) and not force:
        print(f"[INFO] Local file found: {dest} — using local copy.")
        return dest
    print(f"[INFO] Downloading metadata.csv from:\n  {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[INFO] Download complete -> {dest}")
        return dest
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
        if os.path.exists(dest):
            print(f"[INFO] Found existing local file: {dest} — using it.")
            return dest
        raise RuntimeError("Could not download metadata.csv. Please place metadata.csv in the script folder.") from e

def load_metadata(path=LOCAL_FILENAME, nrows=NROWS):
    """Load metadata into a pandas DataFrame. Use nrows (or None)."""
    print(f"[INFO] Loading metadata from {path} (nrows={nrows}) ...")
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"[INFO] Loaded shape: {df.shape}")
    return df

def inspect_df(df, n=5):
    """Print basic inspection info: head, shape, dtypes, missing."""
    print("\n--- FIRST ROWS ---")
    print(df.head(n).to_string())
    print("\n--- SHAPE ---")
    print(df.shape)
    print("\n--- DATATYPES ---")
    print(df.dtypes)
    print("\n--- MISSING VALUES (top 20) ---")
    print(df.isna().sum().sort_values(ascending=False).head(20))

def clean_metadata(df):
    """Clean and prepare DataFrame for analysis."""
    df = df.copy()
    # Ensure title exists; drop rows without title
    if "title" in df.columns:
        before = len(df)
        df = df.dropna(subset=["title"])
        after = len(df)
        print(f"[INFO] Dropped {before-after} rows without title.")
        df["title"] = df["title"].astype(str)
    else:
        raise KeyError("The dataset does not contain a 'title' column.")

    # Fill missing abstract column if present
    if "abstract" in df.columns:
        df["abstract"] = df["abstract"].fillna("").astype(str)
    else:
        df["abstract"] = ""

    # Parse publish_time into datetime
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df["pub_year"] = df["publish_time"].dt.year
    else:
        df["publish_time"] = pd.NaT
        df["pub_year"] = pd.NA

    # Fill missing journal/source fields
    if "journal" in df.columns:
        df["journal"] = df["journal"].fillna("Unknown Journal")
    else:
        df["journal"] = "Unknown Journal"

    if "source_x" in df.columns:
        df["source_x"] = df["source_x"].fillna("Unknown Source")
    else:
        df["source_x"] = "Unknown Source"

    # Word counts
    df["title_word_count"] = df["title"].apply(lambda s: len(str(s).split()))
    df["abstract_word_count"] = df["abstract"].apply(lambda s: len(str(s).split()))

    print("[INFO] Cleaning complete.")
    return df

def most_common_title_words(df, top_n=50):
    """Return top_n frequent words from titles (simple tokenization)."""
    titles = df["title"].dropna().astype(str).str.lower()
    tokens = []
    for t in titles:
        tokens.extend(re.findall(r"\b[a-z]{3,}\b", t))  # words with 3+ letters
    freq = Counter(tokens)
    return freq.most_common(top_n)

def save_plot_publications_by_year(df, out_dir=OUTPUT_DIR):
    """Plot and save publications by year."""
    if df["pub_year"].notna().any():
        counts = df["pub_year"].value_counts().sort_index()
        plt.figure(figsize=(10,5))
        plt.plot(counts.index.astype(int), counts.values, marker="o")
        plt.title("Number of Publications by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of papers")
        plt.grid(True)
        path = os.path.join(out_dir, "papers_by_year.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[SAVE] {path}")
        return path, counts
    else:
        print("[WARN] No valid publication years found; skipping papers_by_year plot.")
        return None, pd.Series(dtype=int)

def save_plot_top_journals(df, out_dir=OUTPUT_DIR, top_n=20):
    """Plot and save top journals."""
    counts = df["journal"].value_counts().head(top_n)
    plt.figure(figsize=(10, max(4, top_n*0.25)))
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.title(f"Top {top_n} Journals by Number of Papers")
    plt.xlabel("Number of papers")
    path = os.path.join(out_dir, "top_journals.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[SAVE] {path}")
    return path, counts

def save_plot_sources_distribution(df, out_dir=OUTPUT_DIR, top_n=20):
    """Plot and save source_x distribution."""
    counts = df["source_x"].value_counts().head(top_n)
    plt.figure(figsize=(10, max(4, top_n*0.25)))
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.title(f"Top {top_n} Sources by Number of Papers")
    plt.xlabel("Number of papers")
    path = os.path.join(out_dir, "top_sources.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[SAVE] {path}")
    return path, counts

def save_wordcloud_from_title_freq(freq_pairs, out_dir=OUTPUT_DIR, filename="title_wordcloud.png"):
    """Create and save a wordcloud from frequency pairs (list of tuples)."""
    if not WORDCLOUD_AVAILABLE:
        print("[WARN] wordcloud not installed — skipping wordcloud generation.")
        return None
    freqs = dict(freq_pairs)
    wc = WordCloud(width=1200, height=600, collocations=False).generate_from_frequencies(freqs)
    path = os.path.join(out_dir, filename)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[SAVE] {path}")
    return path

def generate_report(df, analysis_results, out_file="brief_report.txt"):
    """Write a short text report summarizing results."""
    lines = []
    lines.append("CORD-19 Metadata Analysis Report")
    lines.append(f"Author: {AUTHOR_NAME}")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"Total records analyzed: {len(df)}")
    if "papers_by_year" in analysis_results and not analysis_results["papers_by_year"].empty:
        top_year = int(analysis_results["papers_by_year"].idxmax())
        lines.append(f"Year with most papers: {top_year} ({analysis_results['papers_by_year'].max()} papers)")
    if "top_journals" in analysis_results:
        lines.append("\nTop journals (sample):")
        for j, c in analysis_results["top_journals"].head(5).items():
            lines.append(f" - {j}: {int(c)}")
    if "title_word_freq" in analysis_results:
        lines.append("\nTop title words (sample):")
        for w, c in analysis_results["title_word_freq"][:20]:
            lines.append(f" - {w}: {c}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- publish_time was coerced to datetime; some entries may be missing.")
    lines.append("- Abstracts filled with empty strings for text features.")
    lines.append("- WordCloud generation is optional and requires the 'wordcloud' package.")
    text = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SAVE] {out_file}")
    return out_file

# -----------------------
# Streamlit application
# -----------------------
def streamlit_app(df, analysis_results):
    """Simple Streamlit app showing the findings (will run only if streamlit is installed)."""
    st.title("CORD-19 Metadata Explorer")
    st.markdown(f"**Author:** {AUTHOR_NAME}")

    st.header("Quick report")
    report_path = "brief_report.txt"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.write("Report not found. Run the script first to generate the report.")

    st.header("Visualizations")
    # Show images if they exist
    for label, fname in [
        ("Publications by Year", "papers_by_year.png"),
        ("Top Journals", "top_journals.png"),
        ("Top Sources", "top_sources.png"),
        ("Title WordCloud", "title_wordcloud.png"),
    ]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=label, use_column_width=True)
        else:
            st.write(f"{label} not found (file: {path})")

    st.header("Data sample")
    st.dataframe(df.head(50))

# -----------------------
# Main workflow
# -----------------------
def main(nrows=NROWS, force_download=False):
    # Download (best effort) or use local copy
    try:
        download_metadata(dest=LOCAL_FILENAME, force=force_download)
    except Exception as e:
        print(f"[ERROR] Could not ensure metadata.csv is present: {e}")
        sys.exit(1)

    # Load
    df = load_metadata(path=LOCAL_FILENAME, nrows=nrows)

    # Inspect (small)
    inspect_df(df, n=3)

    # Clean
    df_clean = clean_metadata(df)

    # Analysis & visuals
    analysis_results = {}

    p_path, year_counts = save_plot_publications_by_year(df_clean)
    analysis_results["papers_by_year"] = year_counts

    j_path, journals = save_plot_top_journals(df_clean, top_n=20)
    analysis_results["top_journals"] = journals

    s_path, sources = save_plot_sources_distribution(df_clean, top_n=20)
    analysis_results["top_sources"] = sources

    # title word freq and optional wordcloud
    title_freq = most_common_title_words(df_clean, top_n=200)
    analysis_results["title_word_freq"] = title_freq
    if WORDCLOUD_AVAILABLE:
        wc_path = save_wordcloud_from_title_freq(title_freq, filename="title_wordcloud.png")
        analysis_results["wordcloud_path"] = wc_path
    else:
        print("[INFO] WordCloud not available — skipped.")

    # Report
    report_path = generate_report(df_clean, analysis_results, out_file="brief_report.txt")

    print("[DONE] Analysis complete. Outputs saved in './outputs/' and report 'brief_report.txt'.")

    return df_clean, analysis_results

# -----------------------
# Execute
# -----------------------
if __name__ == "__main__":
    # Run main analysis
    df_clean, analysis_results = main(nrows=NROWS)

    # If streamlit available and this module is executed under Streamlit, run the app UI
    # When running `streamlit run script.py`, streamlit is typically in sys.modules.
    if STREAMLIT_AVAILABLE and "streamlit" in sys.modules:
        # Build UI using the already computed df_clean and analysis_results
        try:
            streamlit_app(df_clean, analysis_results)
        except Exception as e:
            print(f"[ERROR] Streamlit app failed to start: {e}")
            print("You can still view outputs in ./outputs/ and brief_report.txt")
    else:
        print("To view interactive dashboard (optional):")
        print("  pip install streamlit")
        print("  streamlit run CORD19_full_assignment.py")