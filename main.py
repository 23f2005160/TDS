import json, re, hashlib, subprocess, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
import config

# Global in-memory storage for Q4 Data
Q4_DOCS = []
Q4_EMBEDDINGS = {}
Q4_RERANKER = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------------------------------------------------
    # STARTUP: Generate Q4 Data Offline
    # ---------------------------------------------------------
    print(f"Generating Q4 Data for {config.EMAIL}...")
    try:
        subprocess.run(["node", "q4_generate.js", config.EMAIL], check=True, cwd=BASE_DIR)
        print("Q4 Data generated successfully.")
    except Exception as e:
        print(f"Failed to generate Q4 Data: {e}")
        
    # Load Documents
    try:
        import csv
        with open(os.path.join(BASE_DIR, "documents.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert year to int
                row["year"] = int(row["year"])
                Q4_DOCS.append(row)
                
        with open(os.path.join(BASE_DIR, "embeddings.json"), "r", encoding="utf-8") as f:
            embs = json.load(f)
            # Pre-convert to numpy arrays for fast cosine similarity
            for k, v in embs.items():
                Q4_EMBEDDINGS[k] = np.array(v, dtype=np.float32)
                
        with open(os.path.join(BASE_DIR, "reranker_scores.json"), "r", encoding="utf-8") as f:
            Q4_RERANKER.update(json.load(f))
            
        print(f"Loaded {len(Q4_DOCS)} documents, {len(Q4_EMBEDDINGS)} embeddings, {len(Q4_RERANKER)} queries for reranking.")
    except Exception as e:
        print(f"Failed to load Q4 Data: {e}")
        
    yield
    # SHUTDOWN
    pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

HEAD = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}", "Content-Type": "application/json"}
_CACHE = {}

def _ck(*parts):
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()

import asyncio
async def chat(messages, model=None, max_tokens=800, force_json=True, retries=4):
    key = _ck("chat", model, json.dumps(messages, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    body = {"model": model or config.TEXT_MODEL, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            r = await c.post(f"{config.AIPIPE_BASE}/chat/completions",
                             headers=HEAD, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            _CACHE[key] = out
            return out
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")

def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}

# ================= Q3: /q3/answer =================
@app.post("/q3/answer")
async def q3_answer(request: Request):
    body = await request.json()
    question = body.get("question", "")
    chunks = body.get("chunks", [])
    
    prompt = (
        "You are a highly reliable Grounded QA API for medical and legal compliance.\n"
        "Your task is to answer the user's question strictly using ONLY the provided context chunks.\n"
        "1. If the question CANNOT be answered from the chunks, you MUST return:\n"
        "   - answerable: false\n"
        "   - answer: \"I don't know\" (exact match)\n"
        "   - citations: [] (empty array)\n"
        "   - confidence: 0.1\n"
        "2. If it CAN be answered, return:\n"
        "   - answerable: true\n"
        "   - answer: <your grounded answer>\n"
        "   - citations: [<list of ONLY the chunk_ids you used to formulate the answer>]\n"
        "   - confidence: <float between 0.8 and 1.0>\n"
        "NEVER use outside knowledge. Return strictly JSON with exactly these 4 keys.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CHUNKS:\n{json.dumps(chunks, indent=2)}"
    )
    
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o-mini", max_tokens=1000))
        # Ensure answerable logic is strict
        if not out.get("answerable", False) or out.get("confidence", 1.0) <= 0.3:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }
        # ensure citations are a list and valid
        valid_ids = [c["chunk_id"] for c in chunks]
        cites = [c for c in out.get("citations", []) if c in valid_ids]
        return {
            "answer": out.get("answer", "I don't know"),
            "citations": cites,
            "confidence": float(out.get("confidence", 0.9)),
            "answerable": True
        }
    except Exception as e:
        return {"answer": "I don't know", "citations": [], "confidence": 0.1, "answerable": False}

# ================= Q4: /vector-search =================
def cosine_sim(a, b):
    # a and b are numpy arrays
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

@app.post("/vector-search")
async def vector_search(request: Request):
    body = await request.json()
    query_id = body.get("query_id")
    query_vector = np.array(body.get("query_vector", []), dtype=np.float32)
    top_k = body.get("top_k", 10)
    rerank_top_n = body.get("rerank_top_n", 3)
    filters = body.get("filter", {})
    
    # 1. Filter documents
    filtered_docs = []
    for doc in Q4_DOCS:
        match = True
        for key, condition in filters.items():
            if isinstance(condition, dict):
                if "gte" in condition and not (doc.get(key, 0) >= condition["gte"]): match = False
                if "lte" in condition and not (doc.get(key, 0) <= condition["lte"]): match = False
                if "in" in condition and not (doc.get(key) in condition["in"]): match = False
            else:
                # Exact match
                if doc.get(key) != condition: match = False
        if match:
            filtered_docs.append(doc)
            
    # 2. Compute Cosine Similarity
    scored_docs = []
    for doc in filtered_docs:
        doc_id = doc["doc_id"]
        doc_emb = Q4_EMBEDDINGS.get(doc_id)
        if doc_emb is not None:
            sim = cosine_sim(query_vector, doc_emb)
            scored_docs.append({"doc_id": doc_id, "sim": sim})
            
    # 3. Retrieve top_k
    # Sort descending by similarity, tie-break by lexicographically smaller doc_id
    scored_docs.sort(key=lambda x: (-x["sim"], x["doc_id"]))
    top_k_docs = scored_docs[:top_k]
    
    # 4. Re-ranking
    # Re-rank the retrieved top top_k documents using the lookup table reranker_scores[query_id]
    rerank_scores = Q4_RERANKER.get(query_id, {})
    for doc in top_k_docs:
        # If not found in reranker table, assume very low score
        doc["rerank_score"] = rerank_scores.get(doc["doc_id"], -999.0)
        
    # Sort descending by score, tie-break by lexicographically smaller doc_id
    top_k_docs.sort(key=lambda x: (-x["rerank_score"], x["doc_id"]))
    
    # 5. Output
    final_matches = [d["doc_id"] for d in top_k_docs[:rerank_top_n]]
    
    return {"matches": final_matches}

# ================= Q5: GraphRAG Endpoints =================

@app.post("/extract-graph")
async def extract_graph(request: Request):
    body = await request.json()
    chunk_id = body.get("chunk_id", "")
    text = body.get("text", "")
    
    prompt = (
        "You are an expert GraphRAG Entity and Relationship extractor.\n"
        "Extract entities and relationships from the provided text according to these EXACT rules:\n"
        "Allowed Entity Types: Person, Organization, Product, Framework\n"
        "Allowed Relationship Types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED\n\n"
        "Return strictly JSON in this format:\n"
        "{\n"
        "  \"entities\": [{\"name\": \"Entity Name\", \"type\": \"AllowedType\"}],\n"
        "  \"relationships\": [{\"source\": \"Entity1\", \"target\": \"Entity2\", \"relation\": \"ALLOWED_RELATION\"}]\n"
        "}\n\n"
        f"TEXT:\n{text}"
    )
    
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500))
        return {
            "entities": out.get("entities", []),
            "relationships": out.get("relationships", [])
        }
    except Exception:
        return {"entities": [], "relationships": []}

@app.post("/graph-query")
async def graph_query(request: Request):
    body = await request.json()
    question = body.get("question", "")
    graph = body.get("graph", {})
    
    prompt = (
        "You are a GraphRAG multi-hop reasoning agent.\n"
        "Given the knowledge graph provided (entities and relationships), answer the natural language question.\n"
        "You must determine the logical path through the graph to find the answer.\n"
        "Return strictly JSON in this format:\n"
        "{\n"
        "  \"answer\": \"Brief factual answer\",\n"
        "  \"reasoning_path\": [\"Entity1\", \"Entity2\", \"Entity3\"], // The sequence of nodes traversed\n"
        "  \"hops\": 2 // Number of edges traversed (which is len(reasoning_path) - 1)\n"
        "}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GRAPH:\n{json.dumps(graph, indent=2)}"
    )
    
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500))
        path = out.get("reasoning_path", [])
        return {
            "answer": out.get("answer", ""),
            "reasoning_path": path,
            "hops": len(path) - 1 if len(path) > 0 else 0
        }
    except Exception:
        return {"answer": "", "reasoning_path": [], "hops": 0}

@app.post("/community-summary")
async def community_summary(request: Request):
    body = await request.json()
    community_id = body.get("community_id", "")
    entities = body.get("entities", [])
    relationships = body.get("relationships", [])
    
    prompt = (
        f"You are a GraphRAG community summarizer. Summarize the following community of entities and relationships.\n"
        "The summary should be a concise paragraph explaining how these entities are connected and what their overall theme is based on the relationships.\n"
        "Return strictly JSON in this format:\n"
        "{\n"
        f"  \"community_id\": \"{community_id}\",\n"
        "  \"summary\": \"Your summary here.\"\n"
        "}\n\n"
        f"ENTITIES:\n{json.dumps(entities, indent=2)}\n\n"
        f"RELATIONSHIPS:\n{json.dumps(relationships, indent=2)}"
    )
    
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500))
        return {
            "community_id": community_id,
            "summary": out.get("summary", "")
        }
    except Exception:
        return {"community_id": community_id, "summary": ""}
