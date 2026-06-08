import gradio as gr
import hydra
from hydra import initialize, compose
from src.inference import RoboScholarInference

with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="config")

inference = RoboScholarInference(cfg)


def upload_pdf(file):
    if file is None:
        return "No file uploaded."
    n_chunks = inference.index_pdf(file.name)
    filename = file.name.split("/")[-1]
    return f"Indexed {n_chunks} chunks from {filename}"


def ask(question):
    if not question.strip():
        return "", ""
    answer, source = inference.answer(question)
    if source:
        source_str = f"Source: {source['source']}, page {source['page']}"
    else:
        source_str = "No relevant context found in index."
    return answer, source_str


with gr.Blocks(title="RoboScholar") as demo:
    gr.Markdown("# RoboScholar\nAsk questions about robotics papers.")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload a robotics paper (PDF)", file_types=[".pdf"])
            upload_btn = gr.Button("Index Paper")
            upload_status = gr.Textbox(label="Upload status", interactive=False)
            upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=upload_status)

        with gr.Column():
            question_input = gr.Textbox(label="Ask a question", placeholder="What is model predictive control?")
            ask_btn = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            source_output = gr.Textbox(label="Retrieved from", interactive=False)
            ask_btn.click(fn=ask, inputs=question_input, outputs=[answer_output, source_output])

if __name__ == "__main__":
    demo.launch()
