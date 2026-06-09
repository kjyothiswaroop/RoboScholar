import gradio as gr
import hydra
from hydra import initialize, compose
from src.inference import RoboScholarInference

with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="config")

inference = RoboScholarInference(cfg)


def upload_pdf(file):
    if file is None:
        return "No file uploaded.", gr.update()
    filename = file.name.split("/")[-1]
    n_chunks = inference.index_pdf(file.name)
    papers = inference.list_papers()
    dropdown = gr.update(choices=papers, value=filename)
    if n_chunks == 0:
        return f"{filename} already exists in the database.", dropdown
    return f"Indexed {n_chunks} chunks from {filename}", dropdown


def ask(question, paper):
    if not question.strip():
        return "", "", ""
    if not paper:
        return "Select a paper to ask about first.", "", ""
    answer, source, excerpt = inference.answer(question, source=paper)
    if source:
        source_str = f"Source: {source['source']}, page {source['page']}"
    else:
        source_str = "No relevant context found in this paper."
    return answer, source_str, excerpt


with gr.Blocks(title="RoboScholar") as demo:
    gr.Markdown("# RoboScholar\nAsk questions about robotics papers.")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload a robotics paper (PDF)", file_types=[".pdf"])
            upload_btn = gr.Button("Index Paper")
            upload_status = gr.Textbox(label="Upload status", interactive=False)

        with gr.Column():
            paper_select = gr.Dropdown(label="Paper to ask about", choices=inference.list_papers())
            question_input = gr.Textbox(label="Ask a question", placeholder="What is model predictive control?")
            ask_btn = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            source_output = gr.Textbox(label="Retrieved from", interactive=False)
            excerpt_output = gr.Textbox(label="Relevant excerpt", interactive=False, lines=6)

    upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=[upload_status, paper_select])
    ask_btn.click(fn=ask, inputs=[question_input, paper_select], outputs=[answer_output, source_output, excerpt_output])

if __name__ == "__main__":
    demo.launch()
