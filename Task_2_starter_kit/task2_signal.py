import re
import torch


SAMPLE_PROMPT = """Task: Analyze the following news headline about a stock and provide a sentiment score between -{signal_strength} and {signal_strength}, where:
- -{signal_strength} means very negative sentiment
- -{threshold} means neutral negative sentiment
- 0 means neutral sentiment
- {threshold} indicates neutral positive sentiment
- {signal_strength} means very positive sentiment

Do not provide any explanations. Output only a single number in the range of -{signal_strength} to {signal_strength} based on the sentiment of the news. 

News headline: "{news}"

Price Data: "{prices}"

Generate only a single integer value for the sentiment score after the colon. Sentiment score:
"""


def _generate_signal(tokenizer, model, device, news, prices, signal_strength, threshold) -> (float, float):  # type: ignore
    """Generate a sentiment signal using the provided model and tokenizer.

    Args:
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained LLM model.
        device: Device to run computations on (CPU/GPU).
        news (str): News headline for sentiment analysis.
        prices (str): Price data associated with the news.
        signal_strength (float): Maximum range for sentiment scores.
        threshold (float): Threshold for determining sentiment polarity.

    Returns:
        Tuple[float, float]: The sentiment score and the total log probability.
    """
    prompt = SAMPLE_PROMPT.format(signal_strength=signal_strength, threshold=threshold, news=news, prices=prices)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = inputs["input_ids"]
    log_probs = []
    max_new_tokens = 5

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits  # shape: [batch_size, seq_length, vocab_size]

        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Replace NaN, Inf, or negative values with 0
        next_token_probs = torch.where(
            torch.isnan(next_token_probs) | torch.isinf(next_token_probs) | (next_token_probs < 0),
            torch.zeros_like(next_token_probs),
            next_token_probs
        )

        # Normalize the probabilities to ensure valid sampling
        next_token_probs = next_token_probs / next_token_probs.sum(dim=-1, keepdim=True)

        next_token_id = torch.multinomial(next_token_probs, num_samples=1)

        token_log_prob = torch.log(next_token_probs[0, next_token_id[0, 0]])
        log_probs.append(token_log_prob)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    total_log_prob = torch.stack(log_probs).sum()
    output_string = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    sentiment_score = float(match.group(1)) if match else 0

    return sentiment_score, total_log_prob


def _generate_eval_signal(tokenizer, model, device, news, prices, signal_strength, threshold) -> float:
    """Generate sentiment signal specifically for evaluation.

    Args:
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained LLM model.
        device: Device to run computations on (CPU/GPU).
        news (str): News headline for sentiment analysis.
        prices (str): Price data associated with the news.
        signal_strength (float): Maximum range for sentiment scores.
        threshold (float): Threshold for determining sentiment polarity.

    Returns:
        float: The sentiment score.
    """
    prompt = SAMPLE_PROMPT.format(signal_strength=signal_strength, threshold=threshold, news=news, prices=prices)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    sentiment_score = float(match.group(1)) if match else 0

    return sentiment_score


def generate_eval_signal(tokenizer, model, device, news, prices, signal_strength, threshold) -> float:
    """Public-facing function to generate sentiment scores for evaluation.

    Args:
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained LLM model.
        device: Device to run computations on (CPU/GPU).
        news (str): News headline for sentiment analysis.
        prices (str): Price data associated with the news.
        signal_strength (float): Maximum range for sentiment scores.
        threshold (float): Threshold for determining sentiment polarity.

    Returns:
        float: The sentiment score.
    """
    return _generate_eval_signal(tokenizer, model, device, news, prices, signal_strength, threshold)


def generate_signal(tokenizer, model, device, news, prices, signal_strength, threshold) -> float:
    """Public-facing function to generate sentiment scores using the model.

    Args:
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained LLM model.
        device: Device to run computations on (CPU/GPU).
        news (str): News headline for sentiment analysis.
        prices (str): Price data associated with the news.
        signal_strength (float): Maximum range for sentiment scores.
        threshold (float): Threshold for determining sentiment polarity.

    Returns:
        float: The sentiment score.
    """
    return _generate_signal(tokenizer, model, device, news, prices, signal_strength, threshold)
