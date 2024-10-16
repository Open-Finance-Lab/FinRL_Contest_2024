"""
You may modify the signal generation pipeline as you wish.

We use an LLM to generate a sentiment score according to the prompt below. 

You can improve the sentiment analysis here or generate your own signal.
"""

import re
import torch

SAMPLE_PROMPT = """Task: Analyze the following news headline about a stock and provide a sentiment score between -10 and 10, where:
- -10 means very negative sentiment
- -3 means neutral negative sentiment
- 0 means neutral sentiment
- 3 indicates neutral positive sentiment
- 10 means very positive sentiment

Do not provide any explanations. Output only a single number in the range of -10 to 10 based on the sentiment of the news. 

News headline: "{news}"

Price Data: "{prices}"

Generate only a single integer value for the sentiment score after the colon. Sentiment score:
"""


def _generate_signal(tokenizer, model, device, news, prices):
    """Using model forward pass to do backprop"""
    prompt = SAMPLE_PROMPT.format(news=news, prices=prices)
    inputs = tokenizer(prompt, return_tensors="pt")#.to(device)

    generated_ids = inputs["input_ids"]
    log_probs = []
    max_new_tokens = 5

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits  # shape: [batch_size, seq_length, vocab_size]

        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        next_token_id = torch.multinomial(next_token_probs, num_samples=1)

        token_log_prob = torch.log(next_token_probs[0, next_token_id[0, 0]])
        log_probs.append(token_log_prob)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    total_log_prob = torch.stack(log_probs).sum()

    output_string = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    sentiment_score = float(match.group(1)) if match else 0

    return sentiment_score, total_log_prob


def generate_signal(tokenizer, model, device, news, prices):
    return _generate_signal(tokenizer, model, device, news, prices)
