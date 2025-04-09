from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.testset import TestsetGenerator
#from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

'''
# Cargar los documentos desde la carpeta especificada
path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
'''

# Load documents
path="/home/jovyan/Documentos/Docs_pdf"
print("Loading documents:", path)
loader = DirectoryLoader(path, show_progress=True)
docs = loader.load()
print("Documents loaded successfully.\n")

'''
for doc in docs:
    if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
        doc.metadata = {}

    # Ensure every document has a 'summary' property
    if not doc.metadata.get("summary"):
        doc.metadata["summary"] = doc.page_content.strip()[:200]  # Use the first 200 characters as a fallback

    # Ensure a 'headlines' property exists
    if "headlines" not in doc.metadata:
        doc.metadata["headlines"] = []
'''
print(f"Loaded {len(docs)} documents.")
print(f"Sample Document loaded successfully")

# Configurar el modelo Deepseek-R1:8B con Ollama
#llm = Ollama(model="deepseek-r1:8b")
llm = OllamaLLM(model="mistral-f16")

# Envolver el modelo para que funcione con Ragas
generator_llm = LangchainLLMWrapper(llm)  # Opción con Langchain
# generator_llm = LlamaIndexLLMWrapper(llm)  # Opción con LlamaIndex
'''
import pprint
with open("wrapper_info.txt", "w") as f:
    f.write("Atributos de generator_llm:\n")
    pprint.pprint(vars(generator_llm), stream=f)
    f.write("\nAtributos de generator_llm.langchain_llm:\n")
    pprint.pprint(vars(generator_llm.langchain_llm), stream=f)
'''

# Configurar el generador de conjuntos de prueba con embeddings (debes definirlo)
generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# Generar el conjunto de prueba con los documentos cargados
dataset_main = generator.generate_with_langchain_docs(docs, testset_size=15, with_debugging_logs=True)

# Convertir el dataset a pandas
df = dataset_main.to_pandas()
df.to_csv('/home/jovyan/RV_2_14/output/dataset_main.csv', index=False)
print(df)

file_path = '/home/jovyan/RV_2_14/output/dataset_main.csv'

if os.path.exists(file_path):
    print("El archivo se ha generado correctamente.")
else:
    print("El archivo no se ha generado.")

# Configurar el token de autenticación para Ragas
os.environ["RAGAS_APP_TOKEN"] = "apt.44d7-55ca6e4e5ccd-d4a1-906b-0f447f55-80b6b"

# Subir el dataset (asegúrate de que tienes una cuenta válida en Ragas)
dataset.upload()
