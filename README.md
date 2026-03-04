# Groww Mutual Fund FAQ Chatbot (HDFC AMC)

A RAG-based facts-only chatbot that answers questions about HDFC Mutual Fund schemes available on Groww. Built with Streamlit, FAISS, and Google Gemini — **no LangChain**.

## Scope

| Item | Details |
|------|---------|
| **Product** | Groww |
| **AMC** | HDFC Asset Management Company Limited |
| **Schemes** | HDFC Top 100 Fund (Large Cap), HDFC Flexi Cap Fund, HDFC ELSS Tax Saver Fund, HDFC Mid-Cap Opportunities Fund |
| **Sources** | 18 public URLs from HDFC MF, Groww, AMFI, SEBI, CAMS, KFintech |

## Architecture

```
sources.csv          →  scraper.py         →  scraped_data/*.txt
(18 public URLs)        (requests + BS4)      (cleaned text + source metadata)
                                                      │
                                                      ▼
                                              rag_engine.py
                                           (chunk → embed → FAISS index)
                                                      │
                                                      ▼
                                                  app.py
                                           (Streamlit chat UI + Gemini LLM)
```

**No LangChain.** The RAG pipeline is custom-built:
- **Embeddings**: Google Gemini `text-embedding-004`
- **Vector store**: FAISS (`IndexFlatIP` with cosine similarity)
- **LLM**: Google Gemini `gemini-2.0-flash`
- **Scraping**: `requests` + `BeautifulSoup4`

## Setup

### 1. Prerequisites

- Python 3.10+
- A free Google Gemini API key — get one at https://aistudio.google.com/apikey

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a `.env` file (copy from the example):

```bash
cp .env.example .env
```

Then edit `.env` and paste your key:

```
GOOGLE_API_KEY=your-key-here
```

Or enter it directly in the Streamlit sidebar when the app starts.

### 4. Scrape the source pages

```bash
python scraper.py
```

This reads `sources.csv`, scrapes each URL, and saves cleaned text files into `scraped_data/`.

> **Note:** Some pages (especially Groww fund pages) use client-side JS rendering and may return limited content. For those, you can manually create a `.txt` file in `scraped_data/` with the format shown below. The scraper will skip files that already exist.

```
Source: https://example.com/page
Title: Page Title
Category: Scheme Details

<paste the page content here>
```

### 5. Run the chatbot

```bash
streamlit run app.py
```

The first run builds a FAISS index from the scraped data (takes ~30 seconds). Subsequent runs load the cached index instantly.

## Project Structure

```
grow-chatbot/
├── app.py                  # Streamlit chat UI
├── rag_engine.py           # Custom RAG: chunking, embedding, FAISS, generation
├── scraper.py              # Web scraper for sources.csv URLs
├── sources.csv             # 18 public source URLs (HDFC MF, Groww, AMFI, SEBI)
├── requirements.txt        # Python dependencies (no LangChain)
├── .env.example            # API key template
├── .streamlit/
│   └── config.toml         # Streamlit theme (Groww green)
├── scraped_data/           # Scraped text files (created by scraper.py)
├── faiss_index.bin         # FAISS index (created on first app run)
├── chunks.json             # Chunk metadata (created on first app run)
├── sample_qa.md            # 10 sample Q&A pairs
└── README.md               # This file
```

## Key Features

- **Facts-only answers** (≤ 3 sentences) with source citation in every response
- **PII detection**: refuses queries containing PAN, Aadhaar, email, phone numbers
- **Opinion/advice refusal**: politely declines buy/sell/portfolio questions with an educational link
- **No performance claims**: does not compute or compare returns; links to official factsheet
- **Rebuild button**: re-index after scraping new pages without restarting the app

## Known Limitations

- Some JS-rendered pages (Groww fund pages, CAMS, KFintech) may yield thin content when scraped with `requests`. Manual `.txt` files can supplement these.
- Gemini free tier has rate limits (15 requests/minute). Heavy usage may hit throttling.
- Expense ratios and other scheme data change periodically; re-scrape to get the latest.
- The chatbot covers only 4 HDFC schemes. Add more URLs to `sources.csv` and re-scrape to expand.

## Disclaimer

This chatbot provides factual information only, sourced from official public pages of HDFC Mutual Fund, Groww, AMFI, and SEBI. It does **not** provide investment advice. Please consult a SEBI-registered financial advisor for investment decisions. Past performance is not indicative of future results.
