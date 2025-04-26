python -m vllm.entrypoints.openai.api_server --model explodinggradients/Ragas-critic-llm-Qwen1.5-GPTQ

python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --dtype auto --api-key EMPTY
# Lanzar, por ejemplo, Mistral-7B-Instruct
python -m vllm.entrypoints.openai.api_server \
       --model mistralai/Mistral-7B-Instruct-v0.2 \
       --dtype auto --port 8000 --api-key EMPTY

python -m vllm.entrypoints.openai.api_server \
       --model mistralai/Mistral-7B-Instruct-v0.2 \
       --dtype auto --port 8000 --api-key EMPTY

## Lo he conseguido con el mismo modelo que utilizaba en DEEPEVAL
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
    --port 8000 --dtype float16 --max-model-len 4096

### para sabaner si está funcionando: 
curl http://localhost:8000/v1/models

## Herramienta para formatear squemas: pip install outlines
https://github.com/dottxt-ai/outlines


## Verificación básica: uso de GPU con nvidia-smi
# Ejecuta el siguiente comando mientras haces una inferencia con Ollama:
watch -n 1 nvidia-smi

# File "/home/jovyan/RV_2_14/venv/lib/python3.11/site-packages/ragas/testset/persona.py"
class PersonaGenerationPrompt(PydanticPrompt[StringIO, Persona]):
    instruction: str = (
        "Using the provided summary, generate a single persona who would likely interact with or benefit from the content."
        "Provide a short, unique name (only one or two words) and a concise role description."
        #"Include a unique name and a concise role description of who they are."
    )

from difflib import get_close_matches
class PersonaList(BaseModel):
    personas: t.List[Persona]

    def __getitem__(self, key: str) -> Persona:
        # Normalizamos el nombre de búsqueda igual que los nombres en la lista
        key_normalized = key.split(" (")[0].strip()
         # Intento de coincidencia exacta primero
        for persona in self.personas:
            if persona.name == key_normalized:
                return persona

        # Intento de coincidencia aproximada usando get_close_matches
        names = [persona.name for persona in self.personas]
        matches = get_close_matches(key_normalized, names, n=1, cutoff=0.6)

        if matches:
            for persona in self.personas:
                if persona.name == matches[0]:
                    return persona
        raise KeyError(f"No persona found with name '{key_normalized}'")

# File "/home/jovyan/RV_2_14/venv/lib/python3.11/site-packages/ragas/testset/synthesizers/multi_hop/base.py" line 173
response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks, max_tokens=1024
        )

        if response is None or not hasattr(response, "query") or not hasattr(response, "answer"):
            raise ValueError("El modelo no devolvió una respuesta válida con 'query' y 'answer'.")

        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=reference_context,
        )

# File "/home/jovyan/RV_2_14/venv/lib/python3.11/site-packages/ragas/testset/synthesizers/generate.py"
✨ Inserta esto antes de llamar a generate() en generate_with_langchain_docs o generate():

from langchain_core.output_parsers import OutputFixingParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from ragas.testset.synthesizers.single_hop.question_answering import QuestionAnsweringSynthesizer
from ragas.testset.synthesizers.schemas import GeneratedQueryAnswer

base_parser = PydanticOutputParser(pydantic_object=GeneratedQueryAnswer)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=self.llm)
qa_synthesizer = QuestionAnsweringSynthesizer(parser=fixing_parser)
query_distribution = [(qa_synthesizer, 1.0)]  # Usa solo ese sintetizador

return self.generate(
    testset_size=testset_size,
    query_distribution=query_distribution,
    ...
)



print("Personas generadas:", [persona.name for persona in self.persona_list])

sudo apt update 
sudo apt install curl -y

# para permitir la detección de hardware:
sudo apt install pciutils lshw -y 

# Instalar ollama
curl -fsSL https://ollama.com/install.sh | sh

# para usar este comando para verificar qué está ocupando el puerto 11434:
por que me ha salido lo siguiente: (venv) root@jupyter-alberto-2egarciaga:~/RV26# ollama serve
Error: listen tcp 127.0.0.1:11434: bind: address already in use
sudo apt install net-tools -y
sudo netstat -tuln | grep 11434

# puedes usar lsof para identificar qué proceso está ocupando el puerto. Si aún no has instalado lsof, hazlo con:
sudo apt install lsof -y

# Después de instalarlo, usa el siguiente comando para identificar el proceso:
sudo lsof -i :11434

# para matar a los procesos
sudo kill -9 PIDs

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


Start Ollama
ollama serve #is used when you want to start ollama without running the desktop application.

ollama pull llama3.1
ollama pull llama3
ollama pull mistral

ollama run llama3.1:70b
ollama pull llama3.1:70b

ollama pull llama-3.1-70b-versatile

NUMA node(s):                         2
NUMA node0 CPU(s):                    0-23,48-71
NUMA node1 CPU(s):                    24-47,72-95



python3.11 -m venv venv
source venv/bin/activate

ollama run deepseek-r1:8b









pip3 install huggingface-hub
# Instalar el modelo en local
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct --include "original/*" --local-dir Meta-Llama-3.1-70B-Instruct

# Primersimo instalar llamacpp
https://python.langchain.com/docs/integrations/llms/llamacpp/

CPU only installation

pip install --upgrade --quiet llama-cpp-python

Installation with OpenBLAS / cuBLAS / CLBlast
llama.cpp supports multiple BLAS backends for faster processing. Use the FORCE_CMAKE=1 environment variable to force the use of cmake and install the pip package for the desired BLAS backend (source).
Example installation with cuBLAS backend:

CMAKE_ARGS="-DLLAMA_CUBLAS=on"
FORCE_CMAKE=1
pip install llama-cpp-python

IMPORTANT: If you have already installed the CPU only version of the package, you need to reinstall it from scratch. Consider the following command:

CMAKE_ARGS="-DLLAMA_CUBLAS=on"
FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

'''
# 1. Instalar cuda: 
https://developer.nvidia.com/cuda-downloads target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
cat /etc/os-release #Para saber la distribucion
lscpu #Para saber la arquitectura 
En nuestro caso tenemos Download Installer for Linux Ubuntu 22.04 x86_64
ejecutamos los siguientes mandatos
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
export TOGETHER_API_KEY=

# 2. Configurar la variable CUDAToolkit_ROOT:
export CUDAToolkit_ROOT=/usr/local/cuda

# 2.2. Ejercutar el mandato de la instalación
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# 3. Verificar que los drivers de NVIDIA estén instalados:
Asegúrate de que los drivers de NVIDIA estén correctamente instalados y funcionando. Puedes verificar si tu sistema detecta las GPUs y los drivers instalados con el siguiente comando:

nvidia-smi

# 4. Instalar ccache (opcional):
Aunque no es crítico para la instalación, el mensaje sugiere instalar ccache para acelerar futuras compilaciones. Puedes instalarlo con:

sudo apt-get install ccache
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version
# Reinstalamos ninja
sudo apt-get install ninja-build
'''


He cambiado una linea
File "/Users/albertog.garcia/Documents/UPM/PRACTICUM/RAGAS-OLLAMA/venv/lib/python3.12/site-packages/ragas/testset/filters.py", line 60, in filter
output["score"] = sum(output.values()) / len(output.values())

# Parte corregida para evitar ZeroDivisionError
 """Calcula el puntaje basado en los valores de output evitando división por cero"""
if len(output.values()) > 0:
    output["score"] = sum(output.values()) / len(output.values())
else:
    output["score"] = 0  # O algún valor predeterminado


# pip install datasets ragas transformers langchain sentence-transformers
# pip install --upgrade transformers
# pip install -U langchain-community
# pip install huggingface_hub
# huggingface-cli login
# Hugginface token: hf_
# huggingface-cli whoami

# Descargar el modelo directamente de hugginface:
https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct

Command
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct --include "original/*" --local-dir Meta-Llama-3.1-70B-Instruct


# Descargar el modelo Code Llama
https://huggingface.co/TheBloke/CodeLlama-70B-Python-GGUF

I recommend using the huggingface-hub Python library:

pip3 install huggingface-hub

Then you can download any individual model file to the current directory, at high speed, with a command like this:

huggingface-cli download TheBloke/CodeLlama-70B-Python-GGUF codellama-70b-python.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False


# How to download the model desde meta
https://www.llama.com/llama-downloads
Visit the Llama repository in GitHub where instructions can be found in the Llama README
# 1. Install the Llama CLI
In your preferred environment run the command below
Command
pip install llama-stack
# 2.Find models list
See latest available models by running the following command and determine the model ID you wish to download:
Command
llama model list
If you want older versions of models, run the command below to show all the available Llama models:
Command
llama model list --show-all
# 3. Select a model
Select a desired model by running:
Command
llama model download --source meta --model-id  MODEL_ID
# 4. Specify custom URL
Llama 3.1: 405B, 70B & 8B
When the script asks for your unique custom URL, please paste the URL below
URL
https://llama3-1.llamameta.net/*?Policy=

Please save copies of the unique custom URLs provided above, they will remain valid for 48 hours to download each model up to 5 times, and requests can be submitted multiple times. An email with the download instructions will also be sent to the email address you used to request the models.

## #######################################################################################################
# 1.New key created:
lsv2_pt_
key-RAGAS: lsv2_pt_

# 2. Install dependencies
pip install -U langchain langchain-openai

# 3. Configure environment to connect to LangSmith.
pr-gargantuan-nothing-25

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_pt_"
LANGCHAIN_PROJECT="pr-gargantuan-nothing-25"

# 4. Run any LLM, Chat model, or Chain. Its trace will be sent to this project.
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")
