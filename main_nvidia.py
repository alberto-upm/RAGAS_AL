from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.testset import TestsetGenerator
#from langchain_community.embeddings import HuggingFaceEmbeddings
import os
#from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import getpass
import os

from langchain_core.callbacks import BaseCallbackHandler

class DebugCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n🚀 [LLM Start] Se está enviando un prompt al modelo:")
        for i, prompt in enumerate(prompts):
            print(f"\n📤 Prompt {i}:\n{'='*40}\n{prompt}\n{'='*40}")

    def on_llm_end(self, response, **kwargs):
        print("\n✅ [LLM End] Respuesta recibida del modelo:")
        print(response)

    def on_llm_error(self, error, **kwargs):
        print("\n❌ [LLM Error] Ocurrió un error durante la llamada al modelo:")
        print(error)

if not os.getenv("NVIDIA_API_KEY"):
    # Note: the API key should start with "nvapi-"
    os.environ["NVIDIA_API_KEY"] = "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"



'''
# Cargar los documentos desde la carpeta especificada
path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
'''
# Load documents
path="/home/jovyan/Documentos/Docs_pdf"
print("Loading documents:")
loader = DirectoryLoader(path, show_progress=True)
docs = loader.load()
print("Documents loaded successfully.\n")


# Configurar el modelo Deepseek-R1:8B con Ollama
#llm = OllamaLLM(model="deepseek-r1:8b")
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# Envolver el modelo para que funcione con Ragas
generator_llm = LangchainLLMWrapper(llm)  # Opción con Langchain
# generator_llm = LlamaIndexLLMWrapper(llm)  # Opción con LlamaIndex

'''
import pprint
with open("wrapper_info_nvidia.txt", "w") as f:
    f.write("Atributos de generator_llm:\n")
    pprint.pprint(vars(generator_llm), stream=f)
    f.write("\nAtributos de generator_llm.langchain_llm:\n")
    pprint.pprint(vars(generator_llm.langchain_llm), stream=f)
'''

# Configurar el generador de conjuntos de prueba con embeddings (debes definirlo)
generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# Crear la lista de callbacks con tu clase personalizada
#callbacks = [DebugCallbackHandler()]

# Generar el conjunto de prueba con los documentos cargados
#dataset = generator.generate_with_langchain_docs(docs, testset_size=15, callbacks=callbacks)
dataset = generator.generate_with_langchain_docs(docs, testset_size=15)


# Convertir el dataset a pandas
df = dataset.to_pandas()
df.to_csv('/home/jovyan/RV_2_14/output/dataset.csv', index=False)
print(df)

file_path = '/home/jovyan/RV_2_14/output/dataset.csv'

if os.path.exists(file_path):
    print("El archivo se ha generado correctamente.")
else:
    print("El archivo no se ha generado.")

# Configurar el token de autenticación para Ragas
os.environ["RAGAS_APP_TOKEN"] = "apt.44d7-55ca6e4e5ccd-d4a1-906b-0f447f55-80b6b"

# Subir el dataset (asegúrate de que tienes una cuenta válida en Ragas)
dataset.upload()
