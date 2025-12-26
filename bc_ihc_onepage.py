# bc_ihc_onepage.py
# One-page Streamlit app that includes:
#   (A) IHC Image Predictor (feature-based baseline via H-DAB color deconvolution)
#   (B) PubMed RAG (LLM-grounded answers with strict [PMID] citations)
#
# Run:
#   streamlit run bc_ihc_onepage.py
#
# Env (choose one LLM provider):
#   export OPENAI_API_KEY="sk-..."        # for OpenAI
#   # or
#   export ANTHROPIC_API_KEY="..."        # for Anthropic
#   # optional for higher PubMed rate limits:
#   export NCBI_API_KEY="your_ncbi_key"

import os, io, re, math, string, html, json, requests
import numpy as np
import pandas as pd
import streamlit as st
from skimage import io as skio, filters, morphology, exposure
from skimage.color import hed_from_rgb, separate_stains
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# (A) IHC Image Predictor (feature-based baseline)
# -------------------------------
def load_image(file):
    data = file.read()
    arr = skio.imread(io.BytesIO(data))
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.dtype != np.uint8:
        arr = exposure.rescale_intensity(arr, out_range=(0, 255)).astype(np.uint8)
    return arr

def quant_dab(rgb):
    # H-DAB color deconvolution
    hed = separate_stains(rgb, hed_from_rgb)
    dab = -hed[..., 2]
    # Normalize DAB channel [0,1]
    dab = (dab - np.min(dab)) / (np.max(dab) - np.min(dab) + 1e-6)
    # Threshold DAB and clean
    th = filters.threshold_otsu(dab)
    pos = dab > th
    pos = morphology.remove_small_objects(pos, 64)
    pos = morphology.remove_small_holes(pos, 64)
    # Rough nuclei mask from hematoxylin
    hematoxylin = -hed[..., 0]
    denom = np.ptp(hematoxylin) + 1e-6  # NumPy 2.0 safe
    hth = filters.threshold_otsu((hematoxylin - hematoxylin.min()) / denom)
    nuclei = (hematoxylin > hth)
    nuclei = morphology.remove_small_objects(nuclei, 64)
    # Simple features
    pct_pos = 100.0 * pos.mean()
    mean_intensity = float(dab[pos].mean()) if pos.any() else 0.0
    nuclei_density = 1000.0 * nuclei.mean()
    return {
        "Percent_Positive": float(pct_pos),
        "Mean_DAB_Intensity": float(mean_intensity),
        "Nuclei_Density": float(nuclei_density),
    }

def status_from_features(marker, feats):
    pct = feats["Percent_Positive"]
    inten = feats["Mean_DAB_Intensity"]
    # Heuristic thresholds (tune for your lab)
    if marker in ["ER", "PR"]:
        return "Positive" if pct >= 1.0 else "Negative"
    if marker == "Ki67":
        return "High" if pct >= 20.0 else "Low"
    if marker == "HER2":
        # Very rough 0/1+/2+/3+ proxy using area + intensity
        if pct < 2 and inten < 0.2:
            score = 0
        elif pct < 10:
            score = 1
        elif pct < 30:
            score = 2
        else:
            score = 3
        return f"{score}+"
    return "Unknown"

def subtype_from_markers(er, pr, ki67, her2, fish_ratio=None):
    ER_pos = (er == "Positive")
    PR_pos = (pr == "Positive")
    Ki67_high = (ki67 == "High")
    # HER2 positivity rule (IHC 3+ OR IHC 2+ with FISH ≥ 2.0)
    her2pos = (her2 == "3+") or (her2 == "2+" and fish_ratio is not None and fish_ratio >= 2.0)
    if (not ER_pos and not PR_pos and not her2pos):
        return "Triple-Negative"
    if (not ER_pos and not PR_pos and her2pos):
        return "HER2-Enriched"
    if ER_pos and (not her2pos) and (not Ki67_high) and PR_pos:
        return "Luminal A"
    if ER_pos and (not her2pos) and (Ki67_high or (not PR_pos)):
        return "Luminal B (HER2-)"
    if ER_pos and her2pos:
        return "Luminal B (HER2+)"
    return "Other/Indeterminate"

# -------------------------------
# (B) PubMed RAG (LLM-grounded)
# -------------------------------
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

def pubmed_search(query, retmax=20, sort="relevance", mindate=None, maxdate=None):
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax, "sort": sort}
    if mindate:
        params["mindate"] = mindate
    if maxdate:
        params["maxdate"] = maxdate
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("esearchresult", {}).get("idlist", [])

def pubmed_summaries(pmids):
    if not pmids:
        return []
    params = {"db": "pubmed", "retmode": "json", "id": ",".join(pmids)}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{EUTILS}/esummary.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for pid, rec in data.get("result", {}).items():
        if pid == "uids":
            continue
        out.append({
            "pmid": pid,
            "title": rec.get("title", ""),
            "journal": rec.get("fulljournalname", ""),
            "pubdate": rec.get("pubdate", ""),
            "authors": [a.get("name", "") for a in rec.get("authors", []) if isinstance(a, dict)],
        })
    return out

def pubmed_abstracts(pmids):
    if not pmids:
        return {}
    params = {"db": "pubmed", "retmode": "xml", "rettype": "abstract", "id": ",".join(pmids)}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    text = r.text
    abs_map = {pid: "" for pid in pmids}
    for chunk in text.split("<PubmedArticle>"):
        m = re.search(r"<PMID[^>]*>(\d+)</PMID>", chunk)
        if not m:
            continue
        pid = m.group(1)
        abs_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", chunk, flags=re.S)
        clean = " ".join([re.sub("<[^>]+>", "", a).strip() for a in abs_parts]).strip()
        abs_map[pid] = html.unescape(clean)
    return abs_map

def chunk_doc(text, pmid, max_words=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words]).strip()
        if chunk:
            chunks.append({"pmid": pmid, "text": chunk})
    return chunks

def build_corpus(abstracts):
    docs = []
    for pmid, abs_txt in abstracts.items():
        if abs_txt:
            docs.extend(chunk_doc(abs_txt, pmid))
    return docs

def rank_chunks(query, chunks, top_k=12):
    texts = [c["text"] for c in chunks]
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.8)
    X = tfidf.fit_transform(texts + [query])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    idxs = np.argsort(-sims)[:top_k]
    return [(sims[i], chunks[i]) for i in idxs]

# ---- LLM adapters ----
def call_openai(prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=600):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a careful medical research assistant. ONLY use the provided PubMed passages. Add [PMID] after each claim you take from a passage. If information is missing, say so."},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_anthropic(prompt, model="claude-3-5-sonnet-20240620", temperature=0.2, max_tokens=800):
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": "You are a careful medical research assistant. ONLY use the provided PubMed passages. Add [PMID] after each claim you take from a passage. If information is missing, say so.",
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    js = r.json()
    return "".join([b.get("text", "") for b in js.get("content", []) if isinstance(b, dict)])

def call_local(prompt, endpoint="http://localhost:11434/api/generate", model="llama3.1", temperature=0.2, max_tokens=800):
    # Example: Ollama API style
    payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature}, "stream": False}
    r = requests.post(endpoint, data=json.dumps(payload), timeout=120, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    js = r.json()
    return js.get("response", "")

def build_prompt(question, ranked_chunks):
    seen = set()
    lines = []
    used = []
    for score, item in ranked_chunks:
        t = item["text"].strip()
        if t in seen:
            continue
        seen.add(t)
        used.append(item["pmid"])
        lines.append(f"[PMID:{item['pmid']}] {t}")
    context = "\n\n".join(lines)
    instr = (
        "Answer the QUESTION using ONLY the CONTEXT passages. "
        "Cite with [PMID] immediately after each claim. "
        "If missing, say so.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"
    )
    return instr, used

# -------------------------------
# ONE-PAGE UI
# -------------------------------
st.set_page_config(page_title="BC IHC — One-Page Suite", layout="wide")
st.title("BC IHC — One-Page Suite")

with st.sidebar:
    st.header("LLM Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Anthropic", "Local"], index=0)
    if provider == "OpenAI":
        model = st.text_input("OpenAI model", "gpt-4o-mini")
    elif provider == "Anthropic":
        model = st.text_input("Anthropic model", "claude-3-5-sonnet-20240620")
    else:
        model = st.text_input("Local model", "llama3.1")
        endpoint = st.text_input("Local endpoint", "http://localhost:11434/api/generate")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    retmax = st.number_input("Max PubMed results", 5, 100, 20, 1)
    top_k = st.number_input("Top passages for context", 3, 30, 12, 1)

# ---- Section A: IHC Image Predictor ----
with st.expander("🔬 IHC Image Predictor (ER/PR/Ki67/HER2 → Surrogate Subtype)", expanded=True):
    uploaded = st.file_uploader(
        "Upload 1–8 IHC images",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
        key="ihc_uploads",
    )
    if uploaded:
        cols = st.columns(4)
        marker_opts = ["ER", "PR", "Ki67", "HER2"]
        assigned = []
        for i, f in enumerate(uploaded):
            with cols[i % 4]:
                guess = "HER2"
                lname = f.name.lower()
                if "er" in lname:
                    guess = "ER"
                if "pr" in lname:
                    guess = "PR"
                if "ki" in lname or "ki67" in lname:
                    guess = "Ki67"
                if "her2" in lname:
                    guess = "HER2"
                m = st.selectbox(f"Marker: {f.name}", marker_opts, index=marker_opts.index(guess), key=f"marker_{i}")
                assigned.append((f, m))

        rows, er_call, pr_call, ki_call, her2_call = [], None, None, None, None
        for f, m in assigned:
            img = load_image(f)
            feats = quant_dab(img)
            call = status_from_features(m, feats)
            if m == "ER":
                er_call = call
            if m == "PR":
                pr_call = call
            if m == "Ki67":
                ki_call = call
            if m == "HER2":
                her2_call = call
            rows.append({"File": f.name, "Marker": m, **feats, "Call": call})

        df = pd.DataFrame(rows).sort_values("Marker")
        st.dataframe(df, use_container_width=True)

        fish = st.number_input("HER2 FISH ratio (optional; used if HER2 2+)", min_value=0.0, step=0.1, value=0.0, format="%.2f")
        fish_ratio = fish if fish > 0 else None

        if er_call and pr_call and ki_call and her2_call:
            subtype = subtype_from_markers(er_call, pr_call, ki_call, her2_call, fish_ratio)
            st.success(f"**Predicted surrogate subtype:** {subtype}")
        else:
            st.warning("Provide all four markers to compute the surrogate subtype.")

        st.download_button(
            "Download per-image metrics (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="ihc_image_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload IHC images to begin. Filenames like 'case1_ER.jpg' help with auto-assignment.")

# ---- Section B: LLM PubMed RAG ----
with st.expander("📚 LLM Literature Chat — PubMed-grounded (RAG)", expanded=True):
    q = st.text_input("Your question (e.g., 'Ki67 cutoff breast cancer prognosis')", "", key="rag_q")
    col1, col2, col3 = st.columns(3)
    with col1:
        mindate = st.text_input("Min year (optional)", "", key="rag_min")
    with col2:
        maxdate = st.text_input("Max year (optional)", "", key="rag_max")
    with col3:
        sort = st.selectbox("Sort", ["relevance", "pub date", "most recent"], index=0, key="rag_sort")
    go = st.button("Search PubMed & Answer with LLM", key="rag_go")

    if go and q.strip():
        try:
            pmids = pubmed_search(
                q.strip(),
                retmax=int(retmax),
                sort="relevance" if sort == "relevance" else "pub date",
                mindate=(mindate or None),
                maxdate=(maxdate or None),
            )
            meta = pubmed_summaries(pmids)
            abstracts = pubmed_abstracts(pmids)

            table_rows = []
            for rec in meta:
                table_rows.append({
                    "PMID": rec["pmid"],
                    "Title": rec["title"],
                    "Journal": rec["journal"],
                    "Year": (rec["pubdate"] or "").split(" ")[0],
                    "Link": f"https://pubmed.ncbi.nlm.nih.gov/{rec['pmid']}/",
                })
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows))

            corpus = build_corpus(abstracts)
            if not corpus:
                st.warning("No abstracts returned; try changing query or date range.")
            else:
                ranked = rank_chunks(q.strip(), corpus, top_k=int(top_k))
                prompt, used_pmids = build_prompt(q.strip(), ranked)

                # (Fix) Avoid nested expanders: use a checkbox for debug prompt view
                show_prompt = st.checkbox("Show LLM prompt (debug)", value=False, key="rag_prompt_debug")
                if show_prompt:
                    st.code(prompt)

                # Call configured LLM
                if provider == "OpenAI":
                    answer = call_openai(prompt, model=model, temperature=temperature)
                elif provider == "Anthropic":
                    answer = call_anthropic(prompt, model=model, temperature=temperature)
                else:
                    endpoint = st.session_state.get("endpoint", "http://localhost:11434/api/generate")
                    # If using Local, we used the sidebar input; re-read it directly:
                    answer = call_local(prompt, model=model, temperature=temperature)

                st.markdown("### Answer")
                st.write(answer)

                if used_pmids:
                    st.markdown("### Sources")
                    for pid in dict.fromkeys(used_pmids):
                        st.markdown(f"- PMID {pid}: https://pubmed.ncbi.nlm.nih.gov/{pid}/")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your API keys/provider settings. Optionally set NCBI_API_KEY.")
    else:
        st.caption("Configure LLM in the sidebar, enter a query, then run.")

st.caption("Tip: Set environment variables OPENAI_API_KEY or ANTHROPIC_API_KEY. Optional: NCBI_API_KEY for higher PubMed rate limits.")
