from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.llms.haystack_wrapper import HaystackLLMWrapper
from ragas.testset import TestsetGenerator
#from langchain_community.embeddings import HuggingFaceEmbeddings
import os
#from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline


#from llama_index.llms import HuggingFaceInferenceAPI
#from llama_index.legacy.embeddings.langchain import LangchainEmbedding

#from llama_index.llms.huggingface.base import HuggingFaceInferenceAPI

#from llama_index.embeddings import HuggingFaceInferenceAPIEmbedding

from haystack.components.generators import HuggingFaceLocalGenerator

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

# Configurar el modelo Deepseek-R1:8B con Ollama
#llm = Ollama(model="deepseek-r1:8b")
#llm = OllamaLLM(model="mistral-f16")
'''
llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    token="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"
)

llm = HuggingFaceInferenceAPI(
    model_name="tiiuae/falcon-7b-instruct",
    token="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"
)
'''
'''
llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    token="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"
)
'''
'''
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "togethercomputer/llama-2-7b-32k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = HuggingFaceLocalGenerator(
    model=model_name,
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.0,
        "pad_token_id": model.config.eos_token_id
    }
)
'''

# 1. Cargar modelo y tokenizer
model_id = "togethercomputer/llama-2-7b-32k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# 2. Crear pipeline de generación
text_gen_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

# 3. Envolver el pipeline con LangChain
langchain_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# 4. Envolverlo con RAGAS
generator_llm = LangchainLLMWrapper(langchain_llm)


# Envolver el modelo para que funcione con Ragas
#generator_llm = LangchainLLMWrapper(llm)  # Opción con Langchain
#generator_llm = LlamaIndexLLMWrapper(llm)  # Opción con LlamaIndex
#generator_llm = HaystackLLMWrapper(llm)  # Opción con Haystack

# Configurar el generador de conjuntos de prueba con embeddings (debes definirlo)
generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# Generar el conjunto de prueba con los documentos cargados
dataset_main = generator.generate_with_langchain_docs(docs, testset_size=15)

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
