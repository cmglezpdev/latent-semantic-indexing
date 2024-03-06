from query_builder import *
from kulback_leibler_divergence import *
from dataset_processing_LSI import load_data
import gradio as gr
from boolean_model import *


# representacion de los documentos en el espacio semantico latente
U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents = load_data()


def search(query1, query2) -> tuple[str, str]:
    return (
        "\n---------MATCH-------\n\n".join([text for _, text in documents_retrieveral_LSI(query1)]), 
        "\n-------MATCH-------\n".join([text for _, text in boolean_model_retrieveral(query2)])
    )

interface = gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label="Query processing with LSI"), gr.Textbox(label="Query processing with Boolean Model")],
    outputs=[gr.TextArea(label="Documents recoverd with LSI"), gr.TextArea(label="Documents recoverd with Boolean Model")],
    live=True,
    title="Recuperación con Información Semántica Latente",
    description="Escribe una consulta para buscar en el sistema",
)

interface.launch()
