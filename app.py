import os
import uuid
from datetime import datetime, date, time as time_type
from typing import List, Dict, Any, Optional
import time

import streamlit as st
import requests
import numpy as np
import pandas as pd

from pypdf import PdfReader
from docx import Document

try:
    import openai
except ImportError:
    openai = None


# =====================
# CONFIG & UTILITIES
# =====================

st.set_page_config(
    page_title="Agentic Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
EVAL_MODEL_NAME = os.getenv("OPENAI_EVAL_MODEL_NAME", MODEL_NAME)
EMBED_MODEL_NAME = os.getenv("OPENAI_EMBED_MODEL_NAME", "text-embedding-3-small")


def get_openai_client():
    """
    Create an OpenAI client using the new 1.x API.
    Expects OPENAI_API_KEY to be set in Streamlit secrets or environment.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.warning(
            "OPENAI_API_KEY is not set. Please add it to Streamlit secrets or environment variables."
        )
        return None

    if openai is None:
        st.error("openai package is not installed. Please add 'openai' to requirements.txt.")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
    except Exception:
        # Fallback for older openai versions
        openai.api_key = api_key
        client = openai
    return client


def embed_texts(client, texts: List[str]) -> np.ndarray:
    """
    Simple embedding helper that works with both openai 1.x client and legacy.
    Returns an array of shape (len(texts), dim).
    """
    if client is None:
        raise RuntimeError("LLM client not available for embeddings.")

    model = EMBED_MODEL_NAME
    try:
        # new 1.x style
        if hasattr(client, "embeddings"):
            resp = client.embeddings.create(model=model, input=texts)
            vectors = [d.embedding for d in resp.data]
        else:
            # legacy
            resp = client.Embedding.create(model=model, input=texts)
            vectors = [d["embedding"] for d in resp["data"]]
        return np.array(vectors, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Error creating embeddings: {e}")


def split_text_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """
    Naive text chunker: splits on paragraph boundaries and packs into chunks
    up to max_chars. Keeps things simple for demo purposes.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = current + ("\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # if a single paragraph is huge, hard-split
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks


# =====================
# SESSION STATE SETUP
# =====================

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, Any]] = []

    if "patient_db" not in st.session_state:
        # toy in-memory patient database
        st.session_state.patient_db: Dict[str, Dict[str, Any]] = {}

    if "appointment_db" not in st.session_state:
        # doctor schedule (extremely simplified)
        st.session_state.appointment_db: List[Dict[str, Any]] = []

    if "logs" not in st.session_state:
        # LLMOps-style logs
        st.session_state.logs: List[Dict[str, Any]] = []

    if "trace" not in st.session_state:
        # last run multi-agent trace
        st.session_state.trace: List[Dict[str, Any]] = []

    if "rag_index" not in st.session_state:
        # local vector-store style RAG over a tiny curated corpus
        st.session_state.rag_index = None

    if "patient_doc_index" not in st.session_state:
        # vector store for uploaded patient docs
        st.session_state.patient_doc_index = {"chunks": [], "vectors": None}

    if "processed_files" not in st.session_state:
        # track uploaded files we've already embedded, to avoid re-processing
        st.session_state.processed_files: Dict[str, bool] = {}

    if "raw_file_text" not in st.session_state:
        # store raw extracted text per file key
        st.session_state.raw_file_text: Dict[str, str] = {}

    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = "Full RAG (Static + Patient Docs)"


init_session_state()


# =====================
# SIMPLE LOCAL RAG CORPUS (STATIC MED ED)
# =====================

LOCAL_RAG_DOCS = [
    {
        "id": "ckd_basics",
        "title": "Chronic Kidney Disease – Basics",
        "source": "Medline-style educational summary (simulated)",
        "content": (
            "Chronic kidney disease (CKD) is a gradual loss of kidney function over time. "
            "Common causes include diabetes and high blood pressure. Early stages may have "
            "few or no symptoms. Management focuses on controlling blood pressure, blood sugar, "
            "and avoiding medications that can harm the kidneys. A nephrologist is a doctor "
            "specializing in kidney care."
        ),
    },
    {
        "id": "diabetes_type2",
        "title": "Type 2 Diabetes – Overview",
        "source": "WHO-style educational summary (simulated)",
        "content": (
            "Type 2 diabetes is a long-term condition where the body does not use insulin properly. "
            "Symptoms can include increased thirst, frequent urination, and fatigue. "
            "Lifestyle changes such as physical activity, healthy eating, and weight management "
            "are key, along with medications prescribed by a clinician when needed."
        ),
    },
    {
        "id": "hypertension",
        "title": "High Blood Pressure – Risks and Management",
        "source": "Medline-style educational summary (simulated)",
        "content": (
            "High blood pressure (hypertension) often has no symptoms but increases the risk of "
            "heart attack, stroke, kidney disease, and other complications. Management includes "
            "regular monitoring, limiting salt intake, avoiding tobacco, managing stress, and "
            "taking prescribed medications under a clinician’s supervision."
        ),
    },
    {
        "id": "healthy_lifestyle",
        "title": "General Heart-Healthy Lifestyle Advice",
        "source": "Guideline-inspired patient leaflet (simulated)",
        "content": (
            "A heart-healthy lifestyle includes regular physical activity, a balanced diet rich "
            "in fruits, vegetables, and whole grains, avoiding tobacco, limiting alcohol, and "
            "getting adequate sleep. These steps support overall cardiovascular health and can "
            "help manage conditions like hypertension and diabetes."
        ),
    },
]


def build_local_rag_index(client):
    """
    Build an in-memory 'vector store' for LOCAL_RAG_DOCS.
    Stores embeddings and metadata in st.session_state.rag_index.
    """
    if st.session_state.rag_index is not None:
        return st.session_state.rag_index

    texts = [doc["content"] for doc in LOCAL_RAG_DOCS]
    vectors = embed_texts(client, texts)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors_norm = vectors / norms

    st.session_state.rag_index = {
        "docs": LOCAL_RAG_DOCS,
        "vectors": vectors_norm,
    }
    return st.session_state.rag_index


def rag_retrieve(client, query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top-k locally embedded documents for a given query.
    """
    index = build_local_rag_index(client)
    vectors = index["vectors"]
    docs = index["docs"]

    query_vec = embed_texts(client, [query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    scores = vectors @ query_vec
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        d = docs[int(i)].copy()
        d["score"] = float(scores[int(i)])
        results.append(d)
    return results


# =====================
# PATIENT DOCUMENT RAG (UPLOADED FILES)
# =====================

def extract_text_from_upload(uploaded_file) -> str:
    """
    Extract text from an uploaded file (PDF, Excel, Word, text).
    Falls back to best-effort decoding for unsupported types.
    """
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            return "\n".join(pages_text)

        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
            return df.to_csv(index=False)

        elif name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df.to_csv(index=False)

        elif name.endswith(".docx"):
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)

        elif name.endswith(".doc"):
            # Best-effort: treat as text; may not always work for old binary .doc files
            return uploaded_file.read().decode("utf-8", errors="ignore")

        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")

        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error extracting text from {uploaded_file.name}: {e}"


def ensure_patient_doc_index():
    if "patient_doc_index" not in st.session_state or st.session_state.patient_doc_index is None:
        st.session_state.patient_doc_index = {"chunks": [], "vectors": None}


def add_patient_doc_chunks(client, file_key: str, filename: str, source_type: str, text: str):
    """
    Chunk the text, embed, and append to patient_doc_index.
    """
    ensure_patient_doc_index()
    index = st.session_state.patient_doc_index

    chunks = split_text_into_chunks(text, max_chars=800)
    if not chunks:
        return

    try:
        vectors = embed_texts(client, chunks)
    except Exception as e:
        st.error(f"Error embedding chunks for {filename}: {e}")
        return

    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors_norm = vectors / norms

    start_idx = len(index["chunks"])
    new_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{file_key}-{start_idx + i}"
        snippet = chunk_text[:200].replace("\n", " ")
        new_chunks.append(
            {
                "id": chunk_id,
                "file_key": file_key,
                "filename": filename,
                "source_type": source_type,
                "text": chunk_text,
                "snippet": snippet,
            }
        )

    if index["vectors"] is None:
        index["vectors"] = vectors_norm
    else:
        index["vectors"] = np.vstack([index["vectors"], vectors_norm])

    index["chunks"].extend(new_chunks)


def patient_docs_rag_retrieve(client, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from uploaded patient documents for a query.
    """
    ensure_patient_doc_index()
    index = st.session_state.patient_doc_index
    if not index["chunks"] or index["vectors"] is None:
        return []

    query_vec = embed_texts(client, [query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    vectors = index["vectors"]
    scores = vectors @ query_vec
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        c = index["chunks"][int(i)].copy()
        c["score"] = float(scores[int(i)])
        results.append(c)
    return results


def build_patient_context_for_query(client, user_input: str, k: int = 3) -> str:
    """
    Build a short context string from uploaded patient docs relevant to the query.
    Also logs to the trace for observability.
    """
    retrieved = patient_docs_rag_retrieve(client, user_input, k=k)
    if not retrieved:
        return ""

    context_lines = []
    for r in retrieved:
        context_lines.append(
            f"File: {r['filename']}\nSnippet: {r['snippet']}"
        )
    context = "\n\n".join(context_lines)

    st.session_state.trace.append(
        {
            "agent": "patient_doc_retriever",
            "output": {"query": user_input, "retrieved_chunks": retrieved},
        }
    )
    return context


def delete_file_from_index(file_key: str):
    """
    Remove all chunks and vectors associated with a given file_key
    from the patient_doc_index, and clean up associated structures.
    """
    ensure_patient_doc_index()
    index = st.session_state.patient_doc_index

    if not index["chunks"]:
        return

    keep_chunks = []
    keep_vectors = []

    for idx, chunk in enumerate(index["chunks"]):
        if chunk.get("file_key") != file_key:
            keep_chunks.append(chunk)
            if index["vectors"] is not None:
                keep_vectors.append(index["vectors"][idx])

    index["chunks"] = keep_chunks
    if keep_vectors:
        index["vectors"] = np.vstack(keep_vectors)
    else:
        index["vectors"] = None

    # Remove bookkeeping
    st.session_state.processed_files.pop(file_key, None)
    st.session_state.raw_file_text.pop(file_key, None)


# =====================
# AGENT IMPLEMENTATIONS
# =====================

def llm_chat(
    client,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    """
    Wrapper for a chat completion call that works with both openai==1.x and older versions.
    """
    if client is None:
        return "LLM client not available. Please configure OPENAI_API_KEY."

    model = model or MODEL_NAME

    try:
        # New 1.x style
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Legacy style
            resp = client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling LLM: {e}"


def entry_classifier_agent(client, user_input: str) -> str:
    """
    Classifies the user input into high-level intents:
    - onboarding
    - appointment
    - records
    - disease_info
    - report_summary
    - general
    """
    system_prompt = (
        "You are an intent classification agent for a healthcare assistant.\n"
        "You must respond with exactly one of these labels:\n"
        "onboarding, appointment, records, disease_info, report_summary, general.\n"
        "No explanations."
    )
    label = llm_chat(client, system_prompt, user_input).lower()
    label = label.strip()
    allowed = [
        "onboarding",
        "appointment",
        "records",
        "disease_info",
        "report_summary",
        "general",
    ]
    if label not in allowed:
        label = "general"
    return label


def member_verification_agent(client, user_input: str) -> Dict[str, Any]:
    """
    Very simple member verification / patient identification.
    Looks for a patient name pattern like 'for my father John Smith' or 'I am Jane Doe'.
    In a real system, this would integrate with an EMR / patient DB.
    """
    system_prompt = (
        "You are a patient identification agent.\n"
        "Extract patient's full name and age if present.\n"
        "Respond as JSON with keys: name, age, is_new_patient (true/false).\n"
        "If you are unsure, set value to null."
    )
    raw = llm_chat(client, system_prompt, user_input)
    return {
        "raw": raw,
    }


def upsert_patient_record(patient_key: str, data: Dict[str, Any]) -> None:
    """
    Insert or update a simple patient record.
    """
    db = st.session_state.patient_db
    if patient_key not in db:
        db[patient_key] = {}
    db[patient_key].update(data)


def appointment_scheduler_agent(
    client,
    user_input: str,
    patient_key: str,
    specialty: str,
    explicit_datetime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Very simple slot discovery + booking simulator.
    In real life, this would query a Doctor Schedule API.
    """
    # Simulate a simple available slot
    appointment_id = str(uuid.uuid4())[:8]
    slot_time = explicit_datetime or datetime.now().strftime("%Y-%m-%d %H:%M")
    slot = {
        "appointment_id": appointment_id,
        "patient_key": patient_key,
        "specialty": specialty,
        "status": "booked",
        "datetime": slot_time,
        "notes": user_input,
    }
    st.session_state.appointment_db.append(slot)

    system_prompt = (
        "You are an appointment confirmation agent.\n"
        "Generate a short, friendly confirmation message for the patient,\n"
        "including the specialty, simulated appointment time, and any prep instructions."
    )
    confirmation_text = llm_chat(
        client,
        system_prompt,
        f"User query or request: {user_input}\nBooked appointment: {slot}",
    )
    return {"slot": slot, "confirmation": confirmation_text}


def medical_records_agent(client, user_input: str, patient_key: str) -> Dict[str, Any]:
    """
    Allows attendants or patients to add/update history.
    For demo we store raw text in the patient_db under 'history_notes'.
    """
    system_prompt = (
        "You are a medical records structuring agent.\n"
        "Rewrite the user's description into concise, structured patient history notes.\n"
        "Use short paragraphs and bullet points when appropriate."
    )
    structured = llm_chat(client, system_prompt, user_input)

    upsert_patient_record(
        patient_key,
        {
            "last_updated": datetime.now().isoformat(),
            "history_notes": structured,
        },
    )
    return {"structured_notes": structured}


def medical_history_summary_agent(client, patient_key: str) -> str:
    """
    Summarizes a patient's medical history for quick viewing.
    """
    patient = st.session_state.patient_db.get(patient_key)
    if not patient or "history_notes" not in patient:
        return "No medical history found for this patient yet."

    system_prompt = (
        "You are a medical summarization agent.\n"
        "Given the patient's history notes, create a brief summary of key diagnoses,\n"
        "treatments, and relevant alerts.\n"
        "Make it easy for patients and doctors to understand."
    )
    return llm_chat(client, system_prompt, patient["history_notes"])


def disease_info_static_and_patient_agent(client, user_input: str) -> str:
    """
    Use both static educational leaflets and patient-doc context.
    """
    # 1) Retrieve static educational docs
    retrieved_med = rag_retrieve(client, user_input, k=3)

    med_blocks = []
    for doc in retrieved_med:
        med_blocks.append(
            f"Title: {doc['title']}\nSource: {doc['source']}\nContent: {doc['content']}"
        )
    med_context = "\n\n---\n\n".join(med_blocks)

    # 2) Retrieve from uploaded patient docs (if available)
    patient_context = build_patient_context_for_query(client, user_input, k=3)

    system_prompt = (
        "You are a healthcare information assistant.\n"
        "You are given:\n"
        "1) A small library of trusted, educational medical leaflets (guideline-inspired).\n"
        "2) Optional snippets from this specific patient's uploaded records.\n\n"
        "Use ONLY this provided context plus general high-level medical knowledge to answer.\n"
        "Explain information in clear, patient-friendly language.\n"
        "Do NOT provide specific prescriptions or exact dosages.\n"
        "You are not a substitute for a clinician; encourage the user to consult their doctor."
    )
    user_prompt = (
        "User question:\n"
        f"{user_input}\n\n"
        "Educational leaflet context:\n"
        f"{med_context}\n\n"
        "Patient-specific document snippets (may be empty if none were uploaded):\n"
        f"{patient_context}"
    )
    answer = llm_chat(client, system_prompt, user_prompt)

    # For debugging / observability, attach retrieved docs to the trace
    st.session_state.trace.append(
        {
            "agent": "disease_info_retriever",
            "output": {
                "mode": "static+patient_docs",
                "med_docs": retrieved_med,
                "patient_doc_context_used": bool(patient_context),
            },
        }
    )
    return answer


def disease_info_patient_only_agent(client, user_input: str) -> str:
    """
    Use only uploaded patient-doc context, no static educational leaflets.
    """
    patient_context = build_patient_context_for_query(client, user_input, k=5)

    system_prompt = (
        "You are a healthcare information assistant.\n"
        "You are given snippets from this specific patient's uploaded records.\n"
        "Base your explanation primarily on those snippets and high-level medical knowledge.\n"
        "Explain information in clear, patient-friendly language.\n"
        "Do NOT provide specific prescriptions or exact dosages.\n"
        "You are not a substitute for a clinician; encourage the user to consult their doctor."
    )
    user_prompt = (
        "User question:\n"
        f"{user_input}\n\n"
        "Patient-specific document snippets (may be empty if none were uploaded):\n"
        f"{patient_context}"
    )
    answer = llm_chat(client, system_prompt, user_prompt)

    st.session_state.trace.append(
        {
            "agent": "disease_info_retriever",
            "output": {
                "mode": "patient_docs_only",
                "patient_doc_context_used": bool(patient_context),
            },
        }
    )
    return answer


def disease_info_no_rag_agent(client, user_input: str) -> str:
    """
    No RAG context at all; purely general, safe high-level answer.
    """
    system_prompt = (
        "You are a healthcare information assistant.\n"
        "You do not have access to any external documents or patient records.\n"
        "Provide only high-level, general educational information.\n"
        "Explain in clear, patient-friendly language.\n"
        "Do NOT provide specific prescriptions or exact dosages.\n"
        "You are not a substitute for a clinician; encourage the user to consult their doctor."
    )
    user_prompt = f"User question:\n{user_input}"
    answer = llm_chat(client, system_prompt, user_prompt)

    st.session_state.trace.append(
        {
            "agent": "disease_info_retriever",
            "output": {
                "mode": "no_rag",
                "patient_doc_context_used": False,
            },
        }
    )
    return answer


def report_summary_agent(client, user_input: str, rag_mode: str) -> str:
    """
    Summarizes lab or imaging reports in easy-to-understand language.
    Optionally augments with relevant uploaded patient-doc snippets depending on rag_mode.
    """
    patient_context = ""
    if rag_mode != "No RAG (LLM Only)":
        patient_context = build_patient_context_for_query(client, user_input, k=3)

    system_prompt = (
        "You are a medical report summarization assistant.\n"
        "Given a raw clinical report or description, and optional snippets from the patient's "
        "uploaded records, provide:\n"
        "1) a plain-language summary,\n"
        "2) key findings,\n"
        "3) follow-up questions the patient might ask their doctor.\n"
        "You are not a substitute for a licensed physician."
    )
    user_prompt = (
        f"Report or description:\n{user_input}\n\n"
        "Relevant snippets from uploaded patient documents (may be empty):\n"
        f"{patient_context}"
    )
    return llm_chat(client, system_prompt, user_prompt)


# =====================
# LLMOps: EVALUATION & LOGGING
# =====================

def evaluate_response(client, user_query: str, assistant_response: str) -> Dict[str, Any]:
    """
    Lightweight QA-based evaluation.
    Uses a separate LLM call to score the response for helpfulness / safety.
    """
    if client is None:
        return {"score": None, "justification": "No LLM client available for evaluation."}

    system_prompt = (
        "You are an evaluation agent for a healthcare assistant.\n"
        "Rate the assistant's response on a 1-5 scale for helpfulness, correctness, and safety.\n"
        "Return JSON with keys: score (1-5 integer) and justification (1-2 sentences)."
    )
    model = EVAL_MODEL_NAME or MODEL_NAME
    txt = llm_chat(
        client,
        system_prompt,
        f"User query: {user_query}\nAssistant response: {assistant_response}",
        model=model,
        temperature=0.0,
    )
    return {"raw": txt}


def log_interaction(
    user_query: str,
    intent: str,
    patient_key: str,
    agent_outputs: Dict[str, Any],
    assistant_response: str,
    eval_result: Dict[str, Any],
):
    """
    Append an interaction log entry, used for historical view & debugging.
    """
    st.session_state.logs.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_query": user_query,
            "intent": intent,
            "patient_key": patient_key,
            "agent_outputs": agent_outputs,
            "assistant_response": assistant_response,
            "evaluation": eval_result,
        }
    )


# =====================
# ORCHESTRATOR
# =====================

def multi_agent_pipeline(user_input: str) -> str:
    """
    Central orchestrator that wires together agents based on intent.
    Returns the final assistant-facing answer.
    Also populates st.session_state.trace for debugging.
    """
    client = get_openai_client()
    st.session_state.trace = []

    # Step 1: intent classification
    intent = entry_classifier_agent(client, user_input)
    st.session_state.trace.append({"agent": "entry_classifier", "output": intent})

    # Step 2: member verification (simplified)
    member_info = member_verification_agent(client, user_input)
    st.session_state.trace.append({"agent": "member_verification", "output": member_info})

    # For demo, use a generic key; in real app, derive from EMR ID or patient name
    patient_key = "demo_patient"

    rag_mode = st.session_state.get("rag_mode", "Full RAG (Static + Patient Docs)")

    agent_outputs: Dict[str, Any] = {"intent": intent, "member_info": member_info, "rag_mode": rag_mode}
    final_answer = ""

    # Step 3: route to specialized agents
    if intent == "onboarding":
        records_out = medical_records_agent(client, user_input, patient_key)
        agent_outputs["records"] = records_out

        history_summary = medical_history_summary_agent(client, patient_key)
        agent_outputs["history_summary"] = history_summary

        final_answer = (
            "I've started a new patient record and summarized the available history:\n\n"
            f"{history_summary}\n\n"
            "You can now proceed to book an appointment or ask about specific conditions."
        )

    elif intent == "appointment":
        specialty = st.session_state.get("default_specialty", "General Medicine")
        appointment_out = appointment_scheduler_agent(
            client,
            user_input,
            patient_key,
            specialty,
            explicit_datetime=None,
        )
        agent_outputs["appointment"] = appointment_out
        final_answer = appointment_out["confirmation"]

    elif intent == "records":
        records_out = medical_records_agent(client, user_input, patient_key)
        history_summary = medical_history_summary_agent(client, patient_key)
        agent_outputs["records"] = records_out
        agent_outputs["history_summary"] = history_summary
        final_answer = (
            "I've updated the medical record and generated a brief summary:\n\n"
            f"{history_summary}"
        )

    elif intent == "disease_info":
        if rag_mode == "Full RAG (Static + Patient Docs)":
            info = disease_info_static_and_patient_agent(client, user_input)
        elif rag_mode == "Patient Docs Only":
            info = disease_info_patient_only_agent(client, user_input)
        else:
            info = disease_info_no_rag_agent(client, user_input)
        agent_outputs["disease_info"] = info
        final_answer = info

    elif intent == "report_summary":
        summary = report_summary_agent(client, user_input, rag_mode=rag_mode)
        agent_outputs["report_summary"] = summary
        final_answer = summary

    else:  # general
        # Optionally pull patient-doc context even for general queries if rag_mode != No RAG
        patient_context = ""
        if rag_mode != "No RAG (LLM Only)":
            patient_context = build_patient_context_for_query(client, user_input, k=2)

        system_prompt = (
            "You are a friendly healthcare assistant.\n"
            "Answer general health-related questions in a safe, non-diagnostic way.\n"
            "You may use the provided patient-document snippets for additional context "
            "but you are not a substitute for a clinician.\n"
            "Encourage the user to consult a physician for medical decisions."
        )
        user_prompt = (
            f"User question:\n{user_input}\n\n"
            "Relevant snippets from uploaded patient documents (may be empty):\n"
            f"{patient_context}"
        )
        final_answer = llm_chat(client, system_prompt, user_prompt)
        agent_outputs["general"] = final_answer

    # Step 4: evaluation
    eval_result = evaluate_response(client, user_input, final_answer)
    st.session_state.trace.append({"agent": "evaluator", "output": eval_result})

    # Step 5: logging
    log_interaction(
        user_query=user_input,
        intent=intent,
        patient_key=patient_key,
        agent_outputs=agent_outputs,
        assistant_response=final_answer,
        eval_result=eval_result,
    )

    return final_answer


# =====================
# STREAMLIT UI
# =====================

def render_chat_tab():
    st.subheader("🔹 Patient / Caregiver Chat")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Describe your situation or request:",
            height=120,
            placeholder=(
                "Example: 'My 70-year-old father has chronic kidney disease. "
                "I want to book a nephrologist for him. Also, can you summarize "
                "latest treatment methods?'"
            ),
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        assistant_answer = multi_agent_pipeline(user_input.strip())

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input.strip(),
                "time": datetime.now().strftime("%H:%M"),
            }
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": assistant_answer,
                "time": datetime.now().strftime("%H:%M"),
            }
        )

    # display chat history
    for msg in st.session_state.chat_history[-20:]:
        align = "flex-start" if msg["role"] == "assistant" else "flex-end"
        color = "#f0f2f6" if msg["role"] == "assistant" else "#d1e3ff"
        with st.container():
            st.markdown(
                f"""
                <div style="display:flex; justify-content:{align}; margin-bottom:4px;">
                    <div style="background-color:{color}; padding:8px 12px; border-radius:8px; max-width:75%;">
                        <small><b>{msg['role'].capitalize()}</b> • {msg['time']}</small><br/>
                        {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_appointments_tab():
    st.subheader("📅 Appointment Scheduling & Tracking")

    client = get_openai_client()

# Reset form fields on next run
    if st.session_state.get("reset_appointment_form") == True:
        st.session_state.reset_appointment_form = False

    st.markdown("### Book a New Appointment")

    # Appointment Form
    with st.form(key="appointment_form"):
        patient_name = st.text_input("Patient Name", key="appointment_patient_name",
            )
                                     
        specialty = st.selectbox(
            "Specialty",
            ["General Medicine", "Cardiology", "Nephrology", "Endocrinology", "Neurology"],
            index=0,
        )
        preferred_date = st.date_input("Preferred date", value=date.today())
        preferred_time = st.time_input("Preferred time", value=datetime.now().time())
        reason = st.text_area(
            "Reason / Notes",
            value=st.session_state.get("form_reason", ""),
            placeholder="Example: Follow-up for chronic kidney disease.",
        )

        submitted = st.form_submit_button("Schedule Appointment")

    # Handle Submission
    if submitted:
        if client is None:
            st.error("Cannot schedule appointment: OpenAI client not configured.")
        else:
            dt_str = datetime.combine(preferred_date, preferred_time).strftime("%Y-%m-%d %H:%M")
            patient_key = patient_name or "demo_patient"
            user_request = (
                f"Schedule an appointment for {patient_name} with {specialty} on {dt_str}. "
                f"Reason: {reason}"
            )

            result = appointment_scheduler_agent(
                client,
                user_request,
                patient_key,
                specialty,
                explicit_datetime=dt_str,
            )

            if "confirmation" in result:
                st.success("Appointment booked!")
                st.write(result["confirmation"])

                # 1. Queue reset for the next run (to clear form fields)
                st.session_state.reset_appointment_form = True

                # 2. Pause briefly to allow the asynchronous database write to complete
                time.sleep(0.1) # <-- ADDED
                
                # 3. Force an immediate rerun to refresh the display and show the new appointment
                st.rerun() # <-- ADDED
            else:
                st.error("Appointment scheduling failed. Please check the logs.")
            st.success("Appointment booked!")
            st.write(result["confirmation"])
            st.session_state.reset_appointment_form = True
    # Show Existing Appointments
    st.markdown("---")
    st.subheader("Existing Appointments")

    if not st.session_state.appointment_db:
        st.info("No appointments have been booked yet in this demo session.")
        return

    # Display each appointment with Delete button
    for idx, appt in enumerate(st.session_state.appointment_db):
        with st.expander(
            f"{appt['specialty']} • {appt['datetime']} • Status: {appt['status']}",
            expanded=False,
        ):
            st.json(appt)

            # Delete this appointment
            if st.button(f"Delete Appointment", key=f"delete_appt_{idx}"):
                st.session_state.appointment_db.pop(idx)
                st.success("Appointment deleted.")
                st.rerun()

def render_files_tab():
    st.subheader("📂 Patient Files & Search")

    client = get_openai_client()

    st.markdown("### Upload Patient Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, Excel, Word, or text files with patient information.",
        type=["pdf", "xlsx", "xls", "csv", "docx", "doc", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and client is not None:
        for f in uploaded_files:
            file_key = f"{f.name}|{getattr(f, 'size', 'unknown')}"
            if file_key in st.session_state.processed_files:
                continue

            text = extract_text_from_upload(f)
            st.session_state.raw_file_text[file_key] = text
            add_patient_doc_chunks(client, file_key, f.name, "upload", text)
            st.session_state.processed_files[file_key] = True

        st.success(f"Processed {len(st.session_state.processed_files)} unique uploaded file(s).")

    ensure_patient_doc_index()
    index = st.session_state.patient_doc_index
    st.markdown(
        f"**Indexed text chunks from uploaded documents:** {len(index['chunks'])}"
    )

    # Raw extracted text viewer + delete controls
    if st.session_state.raw_file_text:
        st.markdown("---")
        st.markdown("### Raw Extracted Text by File")

        for file_key, text in st.session_state.raw_file_text.items():
            display_name = file_key.split("|")[0]
            chunk_count = sum(
                1 for c in index["chunks"] if c.get("file_key") == file_key
            )

            with st.expander(f"{display_name}  •  Chunks indexed: {chunk_count}"):
                st.text_area(
                    "Extracted text (read-only):",
                    text,
                    height=250,
                    key=f"raw_text_{file_key}",
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Delete from index", key=f"delete_{file_key}"):
                        delete_file_from_index(file_key)
                        st.success(f"Removed {display_name} from RAG index and cleared raw text.")
                        st.rerun()
                with col2:
                    st.caption("Re-upload the file if you want to re-index it.")

    st.markdown("---")
    st.markdown("### Semantic Search Over Patient Files")

    search_query = st.text_input(
        "Ask a question or enter a phrase to search within uploaded patient files:"
    )
    if st.button("Search Files"):
        if client is None:
            st.error("Cannot search files: OpenAI client not configured.")
        else:
            if not index["chunks"]:
                st.info("No uploaded patient documents indexed yet.")
            else:
                results = patient_docs_rag_retrieve(client, search_query, k=5)
                if not results:
                    st.info("No relevant chunks found.")
                else:
                    for r in results:
                        with st.expander(f"{r['filename']} • score={r['score']:.3f}"):
                            st.write(r["snippet"])
                            with st.expander("Full chunk text"):
                                st.write(r["text"])


def render_metrics_tab():
    st.subheader("📊 Model & Agent Metrics")

    logs = st.session_state.logs
    if not logs:
        st.info("No interactions have been logged yet.")
        return

    # simple aggregate metrics
    total = len(logs)
    intents = {}
    for log in logs:
        intents[log["intent"]] = intents.get(log["intent"], 0) + 1

    st.markdown(f"**Total Interactions:** {total}")
    st.markdown("**Intent Distribution:**")
    st.bar_chart({"count": intents})

    st.markdown("---")
    st.markdown("### Detailed Interaction Log")

    for i, log in enumerate(reversed(logs), start=1):
        with st.expander(f"{i}. {log['timestamp']} • Intent: {log['intent']}"):
            st.write("**User Query:**")
            st.write(log["user_query"])

            st.write("**Assistant Response:**")
            st.write(log["assistant_response"])

            st.write("**Agent Outputs (debug):**")
            st.json(log["agent_outputs"])

            st.write("**Evaluation Result (raw):**")
            st.write(log["evaluation"].get("raw", log["evaluation"]))


def render_debug_tab():
    st.subheader("🛠 Debugging Trace Panel")

    if not st.session_state.trace:
        st.info("Run a chat interaction first to see the multi-agent trace.")
        return

    for step in st.session_state.trace:
        with st.expander(f"Agent: {step['agent']}"):
            st.json(step["output"])


def main():
    st.title("🏥 Agentic Healthcare Assistant")
    st.caption(
        "Multi-agent AI workflow for medical task automation: onboarding, appointments, "
        "records, disease info retrieval (local RAG + patient docs), and report summarization.\n\n"
        "⚠️ This demo is for educational purposes only and is **not** a substitute for "
        "professional medical advice, diagnosis, or treatment."
    )
    if st.session_state.get("reset_appointment_form") == True:
        st.session_state.pop("appointment_patient_name", None)
        st.session_state.pop("appointment_reason", None)
        st.session_state.pop("form_patient_name", None)
        st.session_state.pop("form_reason", None)
        st.session_state.pop("appointment_specialty_index", None)
        st.session_state.reset_appointment_form = False

    client = get_openai_client()    
    st.sidebar.markdown("# Settings")

    st.title("🏥 Agentic Healthcare Assistant")

    st.caption(
        "Multi-agent AI workflow for medical task automation: onboarding, appointments, "
        "records, disease info retrieval (local RAG + patient docs), and report summarization.\n\n"
        "⚠️ This demo is for educational purposes only and is **not** a substitute for "
        "professional medical advice, diagnosis, or treatment."
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    st.sidebar.markdown("**LLM Settings**")
    st.sidebar.text_input("Model Name", value=MODEL_NAME, key="model_name_display", disabled=True)
    st.sidebar.text_input("Eval Model", value=EVAL_MODEL_NAME, key="eval_model_display", disabled=True)
    st.sidebar.text_input("Embed Model", value=EMBED_MODEL_NAME, key="embed_model_display", disabled=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("RAG Configuration")

    rag_options = [
        "Full RAG (Static + Patient Docs)",
        "Patient Docs Only",
        "No RAG (LLM Only)",
    ]
    default_mode = st.session_state.get("rag_mode", rag_options[0])
    if default_mode not in rag_options:
        default_mode = rag_options[0]
    rag_mode = st.sidebar.radio(
        "RAG Mode",
        rag_options,
        index=rag_options.index(default_mode),
    )
    st.session_state.rag_mode = rag_mode

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Appointment Settings**")
    st.session_state.default_specialty = st.sidebar.selectbox(
        "Default Specialty for Appointment Requests (chat-based)",
        ["General Medicine", "Cardiology", "Nephrology", "Endocrinology", "Neurology"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Session Controls**")
    if st.sidebar.button("Clear Session State"):
       st.session_state.clear()
       # Only call rerun when the button is pressed!
       st.rerun()
   
    tab_chat, tab_appts, tab_files, tab_metrics, tab_debug = st.tabs(
        [
            "💬 Chat Assistant",
            "📅 Appointments",
            "📂 Patient Files & Search",
            "📊 Metrics & Logs",
            "🛠 Debug Trace",
        ]
    )

    with tab_chat:
        render_chat_tab()

    with tab_appts:
        render_appointments_tab()

    with tab_files:
        render_files_tab()

    with tab_metrics:
        render_metrics_tab()

    with tab_debug:
        render_debug_tab()


if __name__ == "__main__":
    main()
