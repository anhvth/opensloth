import itertools
import os
import itertools

def _escape_html(text: str) -> str:
    """Escape HTML special characters efficiently."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_split_points(parts_mask: list[bool]) -> list[int]:
    """Find split points where trainable/non-trainable sections change."""
    split_points = [0]
    for i in range(1, len(parts_mask)):
        if parts_mask[i] != parts_mask[i - 1]:
            split_points.append(i)
    split_points.append(len(parts_mask))
    return split_points


def _process_token_slice(
    input_ids: list[int], a: int, b: int, tokenizer, max_tokens: int
) -> tuple[str, bool]:
    """Process a slice of tokens and return decoded text and truncation flag."""
    decode_token = input_ids[a:b]

    # Count padding tokens efficiently
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        # Find first padding token from the end
        pad_count = 0
        for token in reversed(decode_token):
            if token == tokenizer.pad_token_id:
                pad_count += 1
            else:
                break

        if pad_count > 0:
            decode_token = decode_token[:-pad_count]
    else:
        pad_count = 0

    is_truncated = b - a > max_tokens
    if is_truncated:
        decode_token = decode_token[:max_tokens]
    text = tokenizer.decode(decode_token, skip_special_tokens=False)

    if is_truncated:
        if pad_count:
            text += f"... padding {pad_count} tokens"
        else:
            text += "... (truncated)"

    return text, is_truncated


def debug_chat_dataloader_for_training(dataloader, tokenizer, n_example=10):
    """
    Debug function to log samples from the training dataloader in an HTML format.
    Outputs to both terminal (with colors) and an HTML file with CSS styling.
    """

    g = iter(dataloader)
    html_path = ".log/dataloader_examples.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # Create HTML file with CSS styling
    with open(html_path, "w") as html_file:
        html_file.write(
            """<!DOCTYPE html>
    <html>
    <head>
        <title>Dataloader Examples</title>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #333; }
            .trainable { background-color: #FFEBCD; color: #333; }
            .context { background-color: #E0FFE0; color: #333; }
            th { background-color: #f2f2f2; }
            th, td { border-color: #ddd; }
        }
        
        @media (prefers-color-scheme: dark) {
            body { background-color: #222; color: #f0f0f0; }
            .trainable { background-color: #664a20; color: #f0f0f0; }
            .context { background-color: #2a5a2a; color: #f0f0f0; }
            th { background-color: #444; color: #f0f0f0; }
            th, td { border-color: #555; }
        }
        
        .trainable, .context { padding: 2px; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid; padding: 8px; text-align: left; }
        h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Dataloader Examples</h1>
        <p>This file contains examples of training data with context and trainable parts.</p>
    """
        )

        max_print_tokens = 20000
        html_parts = []  # Collect HTML parts for batch writing

        for i in range(n_example):
            batch = next(g)
            input_ids = batch["input_ids"][0]
            label_ids = batch["labels"][0]
            generate_html_dataloader_example(tokenizer, max_print_tokens, html_parts, i, input_ids, label_ids)

        # Write all HTML parts at once
        html_file.writelines(html_parts)
        html_file.write("</body>\n</html>")

    print(f"More training debug examples written to {html_path}")

def generate_html_dataloader_example(tokenizer, max_print_tokens, html_parts, i, input_ids, label_ids):
    parts_mask = label_ids >= 0  # True is trainable, False is context

            # Find split points efficiently
    split_points = _get_split_points(parts_mask)

    colored_parts = []
    html_parts.append(f"\n    <h2>Example {i+1}</h2>\n")
    html_parts.append(
                "    <table>\n        <tr><th>Text</th><th>Label</th></tr>\n"
            )

    for a, b in itertools.pairwise(split_points):
        text, is_truncated = _process_token_slice(
                    input_ids, a, b, tokenizer, max_print_tokens
                )
        is_trainable = parts_mask[a]
        token_count = b - a

                # Colored text for terminal
        color_code = "\033[93m" if is_trainable else "\033[92m"
        colored_text = f"{color_code}{text}\033[0m"
        colored_parts.append(colored_text)

                # HTML with CSS classes
        css_class = "trainable" if is_trainable else "context"
        emoji = "🟠" if is_trainable else "🟢"
        label_text = "TRAIN" if is_trainable else "CONTEXT"
        token_suffix = "tokens" if token_count > 1 else "token"
        label = f"{emoji} {label_text} {token_count} {token_suffix}"

                # Escape HTML and add row
        text_escaped = _escape_html(text)
        html_parts.append(
                    f"        <tr>\n"
                    f'            <td><span class="{css_class}">'
                    f"{text_escaped}</span></td>\n"
                    f"            <td>{label}</td>\n"
                    f"        </tr>\n"
                )

    html_parts.append("    </table>\n")

            # Print first example to terminal
    if i == 0:
        colored_output = "".join(colored_parts)
        terminal_msg = f"\n=== EXAMPLE #{i+1} ===\n{colored_output}\n"
        print(terminal_msg)



def print_dataloader_example_short(tokenizer, input_ids, label_ids, max_to_print_per_content=50):
    """
    Print tokenized example to terminal with colors for trainable vs context.
    Shows at most `max_to_print_per_content` tokens per example:
      - If longer, shows first 25 and last 25 tokens.
    """
    parts_mask = label_ids >= 0  # True = trainable, False = context
    split_points = _get_split_points(parts_mask)  # same helper as your HTML code

    colored_parts = []
    for a, b in itertools.pairwise(split_points):
        text, _ = _process_token_slice(input_ids, a, b, tokenizer, max_to_print_per_content)
        is_trainable = parts_mask[a]
        color_code = "\033[93m" if is_trainable else "\033[92m"  # yellow/green
        colored_parts.append(f"{color_code}{text}\033[0m")

    # Join into a single string of tokens
    all_tokens = "".join(colored_parts)

    # If too long, truncate to first 25 + last 25
    if len(colored_parts) > max_to_print_per_content:
        first = "".join(colored_parts[:25])
        last = "".join(colored_parts[-25:])
        all_tokens = f"{first} \033[91m...\033[0m {last}"  # red "..." for truncation

    print(all_tokens)


__all__ = ["debug_chat_dataloader_for_training"]
