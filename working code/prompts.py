from typing import List, Optional, Dict, Any

MAX_DOC_TOKENS = 300

TEMPLATES: Dict[str, Dict[str, Any]] = {
    # P0: Baseline (strong grounding + explicit boundaries, docs first)
    "P0": {
        "system":     "You are a helpful assistant. Use only the provided documents to answer.",
        "doc_prefix": "[Document {i}]:\n{doc}\n\n",
        "pre_docs":   "",  # docs first
        "suffix":     "Question: {question}\nAnswer:",
    },

    # P1: Instruction paraphrase only (structure unchanged vs P0)
    "P1": {
        "system":     "Answer using only the provided documents. Do not use outside knowledge.",
        "doc_prefix": "[Document {i}]:\n{doc}\n\n",  # same boundaries as P0
        "pre_docs":   "",
        "suffix":     "Question: {question}\nAnswer:",
    },

    # P2: Boundary tokenization change only (instruction unchanged vs P0)
    "P2": {
        "system":     "You are a helpful assistant. Use only the provided documents to answer.",
        "doc_prefix": "Passage {i}:\n{doc}\n\n",  # changed boundary tokens
        "pre_docs":   "",
        "suffix":     "Question: {question}\nAnswer:",
    },

    # P3: No system grounding (tests dependence on explicit instruction)
    "P3": {
        "system":     "",
        "doc_prefix": "[Document {i}]:\n{doc}\n\n",  # keep boundaries like P0
        "pre_docs":   "",
        "suffix":     "Question: {question}\nAnswer:",
    },

    # P4: Question-first ordering (position sensitivity)
    "P4": {
        "system":     "You are a helpful assistant. Use only the provided documents to answer.",
        "doc_prefix": "[Document {i}]:\n{doc}\n\n",
        "pre_docs":   "Question: {question}",  # question BEFORE docs
        "suffix":     "Answer:",               # answer cue after docs
    },

    # P5: Structured wrapper format (format robustness)
    "P5": {
        "system":     "You are a helpful assistant. Use only the provided documents to answer.",
        "doc_prefix": '<doc id="{i}">{doc}</doc>\n',
        "pre_docs":   "",
        "suffix":     "<question>{question}</question>\n<answer>",
    },
}

# Frozen decoding — identical across ALL scripts
DECODE_CFG = {
    "max_new_tokens":     64,
    "do_sample":          False,
    # Note: temperature/top_p are ignored when do_sample=False, but we keep them fixed.
    "temperature":        1.0,
    "top_p":              1.0,
    "repetition_penalty": 1.0,
}

REFUSE_STRING = "Not enough information."
CONFLICT_STRING = "Documents contain conflicting information."


def render_prompt(
    template_id: str,
    question: str,
    docs: List[str],
    instruction: Optional[str] = None,
    tokenizer=None,
) -> str:
    """
    Render a prompt using template_id, optionally truncating docs by token count.

    Parameters
    ----------
    template_id : str
        One of {"P0","P1","P2","P3","P4","P5"}.
    question : str
        User question.
    docs : List[str]
        Retrieved documents/chunks.
    instruction : Optional[str]
        Extra instruction, e.g. mode routing instruction or triage policy.
    tokenizer : optional
        HF tokenizer; if provided, docs are truncated to MAX_DOC_TOKENS tokens.

    Returns
    -------
    str
        Fully rendered prompt string.
    """
    if template_id not in TEMPLATES:
        raise KeyError(f"Unknown template_id={template_id}. Valid: {sorted(TEMPLATES.keys())}")

    t = TEMPLATES[template_id]

    doc_block = ""
    for i, doc in enumerate(docs, 1):
        text = doc
        if tokenizer is not None:
            ids = tokenizer(doc, add_special_tokens=False).input_ids
            if len(ids) > MAX_DOC_TOKENS:
                text = tokenizer.decode(ids[:MAX_DOC_TOKENS], skip_special_tokens=True)
        doc_block += t["doc_prefix"].format(i=i, doc=text)

    pre_docs = t.get("pre_docs", "")
    pre_docs = pre_docs.format(question=question) if pre_docs else ""

    parts = [
        t.get("system", ""),
        instruction or "",
        pre_docs,
        doc_block.rstrip(),
        t["suffix"].format(question=question),
    ]
    # Always "\n\n" separator; filter empty strings
    parts = [p for p in parts if p]
    return "\n\n".join(parts)
