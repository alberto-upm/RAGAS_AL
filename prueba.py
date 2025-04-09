from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import time
import uuid
import asyncio

# =============================================================================
# Clase wrapper para formatear la respuesta del modelo Ollama de forma similar
# a la que devuelve ChatNVIDIA. Se añade la implementación de agenerate_prompt
# para compatibilidad con el resto del sistema.
# =============================================================================
class OllamaWrapperWithFormat:
    def __init__(self, ollama_llm):
        self.ollama_llm = ollama_llm  # Instancia del modelo Ollama

    def __call__(self, prompt, **kwargs):
        # Llamamos al modelo Ollama con el prompt y argumentos adicionales
        respuesta_bruta = self.ollama_llm(prompt, **kwargs)
        
        # Obtenemos la marca de tiempo actual
        ahora = int(time.time())
        
        # Generamos un ID único para esta respuesta
        id_unico = "chatcmpl-" + str(uuid.uuid4())
        
        # Formateamos la respuesta para que tenga la misma estructura que la API de NVIDIA
        response_formatted = {
            "id": id_unico,
            "object": "chat.completion",
            "created": ahora,
            "model": self.ollama_llm.model,  # Se asume que el modelo tiene este atributo
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": respuesta_bruta  # Contenido generado por el modelo
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": None,       # Puedes implementar un contador de tokens si lo requieres
                "completion_tokens": None,
                "total_tokens": None
            }
        }
        return response_formatted

    # =============================================================================
    # Método asíncrono requerido por algunos componentes que llaman a agenerate_prompt.
    # Se delega la llamada a __call__ usando asyncio.to_thread para ejecutarlo en un hilo.
    # =============================================================================
    async def agenerate_prompt(self, prompt, **kwargs):
        return await asyncio.to_thread(self.__call__, prompt, **kwargs)

# =============================================================================
# Cargar documentos desde la carpeta especificada
# =============================================================================
path = "/home/jovyan/Documentos/Docs_pdf"
print("Cargando documentos:")
loader = DirectoryLoader(path, show_progress=True)
docs = loader.load()
print("Documentos cargados correctamente.\n")

# =============================================================================
# Asegurar que cada documento tenga la propiedad 'headlines' en su metadata.
# Esto evita que la transformación HeadlineSplitter falle.
# =============================================================================
for doc in docs:
    if 'headlines' not in doc.metadata:
        # Utilizamos la primera línea del contenido del documento como headline.
        lines = doc.page_content.splitlines()
        if lines:
            doc.metadata['headlines'] = lines[0]
        else:
            doc.metadata['headlines'] = "Sin título"

# =============================================================================
# Configurar el modelo de Ollama (en este ejemplo, 'mistral-f16')
# =============================================================================
llm = OllamaLLM(model="mistral-f16")

# Envolver el modelo de Ollama con nuestro wrapper personalizado para dar el formato deseado
wrapped_llm = OllamaWrapperWithFormat(llm)

# Envolver el modelo para que funcione con Ragas
generator_llm = LangchainLLMWrapper(wrapped_llm)

# =============================================================================
# Configurar el generador de conjuntos de prueba usando embeddings de HuggingFace
# =============================================================================
generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# =============================================================================
# Generar el conjunto de prueba utilizando los documentos cargados
# =============================================================================
dataset_main = generator.generate_with_langchain_docs(docs, testset_size=15)

# Convertir el dataset a un DataFrame de pandas y guardarlo en CSV
df = dataset_main.to_pandas()
df.to_csv('/home/jovyan/RV_2_14/output/dataset_main.csv', index=False)
print(df)

# Verificar si el archivo se ha generado correctamente
file_path = '/home/jovyan/RV_2_14/output/dataset_main.csv'
if os.path.exists(file_path):
    print("El archivo se ha generado correctamente.")
else:
    print("El archivo no se ha generado.")

# =============================================================================
# Configurar el token de autenticación para Ragas y subir el dataset
# =============================================================================
os.environ["RAGAS_APP_TOKEN"] = "api ragas"
dataset_main.upload()
