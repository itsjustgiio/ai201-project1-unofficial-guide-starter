"""Gradio interface for the grounded CCNY unofficial guide RAG system."""

from __future__ import annotations

import gradio as gr

from query import ask


def handle_query(question: str) -> tuple[str, str]:
    try:
        result = ask(question)
    except Exception as exc:
        return f"Error: {exc}", ""

    sources = "\n".join(f"- {source}" for source in result["sources"])
    if not sources:
        sources = "No sources retrieved."
    return result["answer"], sources


with gr.Blocks(title="CCNY Unofficial Guide") as demo:
    gr.Markdown("# CCNY Unofficial Guide")

    question = gr.Textbox(label="Your question", lines=2)
    ask_button = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=6)

    ask_button.click(handle_query, inputs=question, outputs=[answer, sources])
    question.submit(handle_query, inputs=question, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
