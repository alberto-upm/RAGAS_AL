import json
import os
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA

def load_documents(path):
    print("Cargando documentos desde:", path)
    loader = DirectoryLoader(path, show_progress=True)
    docs = loader.load()
    print("Documentos cargados correctamente.\n")
    return docs

def monitor_output(generator_llm, prompt):
    """
    Llama al modelo utilizando el método generate_text y monitorea la salida.
    Intenta interpretar la respuesta como JSON para listar los campos que se devuelven.
    """
    print(f"\nLlamando al modelo con el prompt: {prompt}")
    # Se utiliza generate_text en lugar de llamar al wrapper directamente.
    response = generator_llm(prompt)
    
    # La salida es un objeto LLMResult. Se asume que la respuesta principal está en la primera generación.
    try:
        # Se accede al texto de la primera generación
        raw_response = response.generations[0][0].text
        print("Respuesta cruda recibida:")
        print(raw_response)
        
        # Intentar interpretar la respuesta como JSON
        response_json = json.loads(raw_response)
        print("\nRespuesta en formato JSON:")
        print(json.dumps(response_json, indent=4, ensure_ascii=False))
        print("Campos de la respuesta:", list(response_json.keys()))
    except Exception as e:
        print("\nLa respuesta no es un JSON válido o ocurrió un error al parsearla.")
        print("Mostrando la respuesta cruda y el error:")
        print(raw_response)
        print("Error:", e)
    
    return response

def main():
    # Ruta de los documentos (ajusta la ruta según corresponda)
    #docs_path = "/home/jovyan/Documentos/Docs_pdf"
    #docs = load_documents(docs_path)
    
    # Definir un prompt de prueba (puedes modificarlo según tus necesidades)
    test_prompt = "Resume el contenido de los documentos cargados."
    
    # --- Monitorización del modelo a través del servidor local (Ollama) ---
    print("\n--- Monitorizando modelo a través del servidor local (Ollama) ---")
    # Se configura el modelo local (ejemplo: mistral-f16)
    ollama_llm = OllamaLLM(model="mistral-f16")
    generator_ollama = LangchainLLMWrapper(ollama_llm)
    response_ollama = monitor_output(generator_ollama, test_prompt)
    
    # --- Monitorización del modelo a través de la API de Nvidia ---
    print("\n--- Monitorizando modelo a través de la API de Nvidia ---")
    # Configurar la API key de Nvidia (asegúrate de tener la clave correcta)
    if not os.getenv("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "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"
    # Se configura el modelo por API (ejemplo: mistralai/mixtral-8x7b-instruct-v0.1)
    nvidia_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
    generator_nvidia = LangchainLLMWrapper(nvidia_llm)
    response_nvidia = monitor_output(generator_nvidia, test_prompt)

if __name__ == "__main__":
    main()
