"""
You may modify the sentiment generation pipeline as you wish. 

We will import the generate sentiment function from this file and use that for testing. Should this file not work, we will use out generate sentiment function instead.
"""

import re

SAMPLE_PROMPT = """Task: Analyze the following news headline about a stock and provide a sentiment score between -2 and 2, where:
- -2 means very negative sentiment
- 0 means neutral sentiment
- 2 means very positive sentiment

Do not provide any explanations. Output only a single number in the range of -2 to 2 based on the sentiment of the news.

News headline: "{news}"

Price Data: "{prices}"

Generate only a single integer value for the sentiment score after the colon. Sentiment score:
"""


def _generate_sentiment(tokenizer, model, device, news, prices):
    prompt = SAMPLE_PROMPT.format(news=news, prices=prices)

    # using news signals, prompt model for a scaled sentiment scorea
    input = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id
    )
    output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    return float(match.group(1)) if match else 0


def generate_sentiment(tokenizer, model, device, news, prices):
    return generate_sentiment(tokenizer, model, device, news, prices)
