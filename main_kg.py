from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.testset import TestsetGenerator
#from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

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

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms

kg = KnowledgeGraph()
print("kg 1:", kg)

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
print("kg 2:", kg)


# Configurar el modelo Deepseek-R1:8B con Ollama
llm = OllamaLLM(model="deepseek-r1:8b")
#llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# Envolver el modelo para que funcione con Ragas
generator_llm = LangchainLLMWrapper(llm)  # Opción con Langchain
# generator_llm = LlamaIndexLLMWrapper(llm)  # Opción con LlamaIndex

# Configurar el generador de conjuntos de prueba con embeddings (debes definirlo)
generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)


# define your LLM and Embedding Model
# here we are using the same LLM and Embedding Model that we used to generate the testset
transformer_llm = generator_llm
embedding_model = generator_embeddings

trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, trans)

kg.save("/home/jovyan/RV_2_14/knowledge_graph.json")
loaded_kg = KnowledgeGraph.load("/home/jovyan/RV_2_14/knowledge_graph.json")
print("loaded_kg", loaded_kg)

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=loaded_kg)

from ragas.testset.synthesizers import default_query_distribution

query_distribution = default_query_distribution(generator_llm)

# Generar el conjunto de prueba con los documentos cargados
#dataset = generator.generate_with_langchain_docs(docs, testset_size=15)

dataset = generator.generate(testset_size=10, query_distribution=query_distribution)

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
