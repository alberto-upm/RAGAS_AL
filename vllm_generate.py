# -*- coding: utf-8 -*-
"""
main.py – Generación de dataset sintético con Ragas
Versión: vLLM (reemplaza Ollama por ChatOpenAI)
Autor: alberto-upm
"""

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI              # ← nuevo
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------
# CONFIGURACIÓN DEL SERVIDOR vLLM
# ------------------------------------------------------------------
VLLM_BASE_URL = "http://localhost:8000/v1"           # Endpoint del api_server
#VLLM_MODEL    = "mistralai/Mistral-7B-Instruct-v0.2" # Modelo que arrancaste
#VLLM_MODEL    = "explodinggradients/Ragas-critic-llm-Qwen1.5-GPTQ"
VLLM_MODEL    = "NousResearch/Meta-Llama-3-8B-Instruct"
OPENAI_API_KEY = "EMPTY"                             # Clave dummy para vLLM

# ------------------------------------------------------------------
# CARGA DE DOCUMENTOS
# ------------------------------------------------------------------
'''
DOCS_PATH = "/home/jovyan/Documentos/Docs_pdf"
print(f" Cargando documentos desde: {DOCS_PATH}")

loader = DirectoryLoader(DOCS_PATH, show_progress=True)
docs   = loader.load()

print(f"📂 Documentos cargados: {len(docs)}\n")
'''
# ---------------------- Carga y pre-split PDFs ----------------------
DOCS_PATH = "/home/jovyan/Documentos/Docs_pdf"
print(f"📂 Leyendo PDFs en: {DOCS_PATH}")
raw_docs = DirectoryLoader(DOCS_PATH, show_progress=True).load()

pre_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,           # ≈2 k tokens
    chunk_overlap=200,
    add_start_index=True,
)
docs = pre_splitter.split_documents(raw_docs)
print(f"📄 Chunks generados tras split inicial: {len(docs)}")

# ------------------------------------------------------------------
# INICIALIZACIÓN DEL LLM (vLLM + LangChain) 🦙
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model= VLLM_MODEL,
    openai_api_key= OPENAI_API_KEY,
    openai_api_base= VLLM_BASE_URL,
    max_tokens= 2048,
    temperature= 0.0,
)

# Ragas necesita el wrapper para convertir el LLM de LangChain
generator_llm = LangchainLLMWrapper(llm)

# ------------------------------------------------------------------
# EMBEDDINGS (sin cambios)
# ------------------------------------------------------------------
generator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------------
# GENERACIÓN DEL DATASET
# ------------------------------------------------------------------
generator = TestsetGenerator(
    llm= generator_llm,
    embedding_model= generator_embeddings,
)

size = 15
dataset_main = generator.generate_with_langchain_docs(
    docs,
    testset_size= size,
    with_debugging_logs= True,
)
print(f"✅ Preguntas generadas: {size}")
# ------------------------------------------------------------------
# PERSISTENCIA EN CSV
# ------------------------------------------------------------------
import pandas as pd
OUTPUT_CSV = "/home/jovyan/RV_2_14/output/dataset_vllm_generate.csv"

df = dataset_main.to_pandas()
df.to_csv(OUTPUT_CSV, index=False)
print(df)

if os.path.exists(OUTPUT_CSV):
    print(f"💾 Dataset sintético guardado en {OUTPUT_CSV}")
else:
    print("✗ No se ha generado el CSV.")

# ------------------------------------------------------------------
# (OPCIONAL) SUBIDA A Ragas Cloud
# ------------------------------------------------------------------
os.environ["RAGAS_APP_TOKEN"] = "RAGAS_APP_TOKEN"
dataset_main.upload()
