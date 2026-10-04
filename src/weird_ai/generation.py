

import torch


def text_to_token_ids(text, tokenizer):

    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze(0)

    return tokenizer.decode(token_ids.tolist())


def generate_text_simple(model, input_ids, max_new_tokens, context_size):
   

    for _ in range(max_new_tokens):

        # Keep only the most recent context_size tokens
        idx_cond = input_ids[:, -context_size:]

        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Get the logits for the final token
        logits = logits[:, -1, :]

        # Select the most likely next token
        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

        # Append the new token
        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1
        )

    return input_ids


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    context_size,
    max_new_tokens=50
):

    model.eval()

    input_ids = text_to_token_ids(
        start_context,
        tokenizer
    ).to(device)

    token_ids = generate_text_simple(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=context_size
    )

    text = token_ids_to_text(
        token_ids,
        tokenizer
    )

    print(text)