import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model once when server starts
MODEL_NAME = "distilgpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


def compute_surprisal(text: str):
    """
    Compute token-level surprisal using Shannon formula:
    I(x) = -log P(x)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[1:]
    surprisal_scores = []

    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i]
        prob = probs[0, i - 1, token_id]
        surprisal = -torch.log(prob + 1e-9)
        surprisal_scores.append(surprisal.item())

    return list(zip(tokens, surprisal_scores))


def compress_prompt(text: str, prune_ratio: float = 0.3):
    """
    Remove lowest-surprisal tokens based on prune_ratio
    """
    token_scores = compute_surprisal(text)

    # Sort tokens by surprisal (ascending = low information first)
    sorted_scores = sorted(token_scores, key=lambda x: x[1])
    cutoff = int(len(sorted_scores) * prune_ratio)

    low_info_tokens = set(token for token, _ in sorted_scores[:cutoff])

    pruned_tokens = [
        token for token, score in token_scores
        if token not in low_info_tokens
    ]

    compressed_text = tokenizer.convert_tokens_to_string(pruned_tokens)

    stats = {
        "original_token_count": len(tokenizer.encode(text)),
        "compressed_token_count": len(tokenizer.encode(compressed_text)),
        "compression_ratio": round(
            1 - (len(tokenizer.encode(compressed_text)) /
                 max(1, len(tokenizer.encode(text)))), 3
        )
    }

    return compressed_text, stats
