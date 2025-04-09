conda install -c conda-forge libmagic -y
pip install --prefer-binary "unstructured[pdf]"

pip install --prefer-binary "unstructured[md]"

pip install --upgrade transformers
pip install "accelerate>=0.26.0"

pip install -U langchain-ollama
pip install -U langchain-huggingface
pip install langchain-nvidia-ai-endpoints
pip install --upgrade torch torchvision
pip install text-generation-inference

pip install llama-index-llms-huggingface
pip install llama-index-llms-huggingface-api
pip install "transformers[torch]" "huggingface_hub[inference]"
pip install llama-index

pip install --upgrade langchain langchain-core

pip install haystack-ai  #para HuggingFaceLocalGenerator
pip install einops