import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# =========================
# MODEL LOADING (LOAD ONCE)
# =========================

MODEL_NAME = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


# =========================
# SHANNON SURPRISAL
# I(x) = -log P(x)
# =========================

def compute_surprisal(text: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[1:]
    scores = []

    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i]
        prob = probs[0, i - 1, token_id]
        surprisal = -torch.log(prob + 1e-9)
        scores.append(surprisal.item())

    return list(zip(tokens, scores))


# =========================
# PROTECTION RULES
# =========================

def is_protected_token(token: str):
    clean = token.strip()

    # Protect numbers
    if clean.isdigit():
        return True

    # Protect capitalized words (possible proper nouns)
    if len(clean) > 0 and clean[0].isupper():
        return True

    # Protect negations
    if clean.lower() in ["not", "no", "never"]:
        return True

    # Protect important structural words
    if clean.lower() in ["if", "then", "else", "because"]:
        return True

    return False


# =========================
# ENTROPY COMPRESSION ENGINE
# =========================

def compress_prompt(text: str, prune_ratio: float = 0.3):

    if not text.strip():
        return text, {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_ratio": 0.0
        }

    token_scores = compute_surprisal(text)

    # Sort by surprisal (lowest first = low info)
    sorted_tokens = sorted(token_scores, key=lambda x: x[1])

    cutoff = int(len(sorted_tokens) * prune_ratio)

    low_info_tokens = set()

    for token, score in sorted_tokens:
        if len(low_info_tokens) >= cutoff:
            break
        if not is_protected_token(token):
            low_info_tokens.add(token)

    pruned_tokens = [
        token for token, score in token_scores
        if token not in low_info_tokens
    ]

    compressed_text = tokenizer.convert_tokens_to_string(pruned_tokens)

    original_tokens = len(tokenizer.encode(text))
    compressed_tokens = len(tokenizer.encode(compressed_text))

    compression_ratio = 1 - (compressed_tokens / max(1, original_tokens))

    stats = {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": round(compression_ratio, 3)
    }

    return compressed_text, stats
