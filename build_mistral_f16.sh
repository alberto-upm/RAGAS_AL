#!/bin/bash

set -e  # Salir si hay error

# Nombre del modelo en Hugging Face (sin cuantizar)
HF_REPO="mistralai/Mistral-7B-Instruct-v0.1"
MODEL_NAME="mistral-7b-instruct"
GGUF_NAME="mistral-f16.gguf"
OLLAMA_MODEL_NAME="mistral-f16"
: << 'EOF'
EOF
echo "=== 1. Clonando llama.cpp y descargando scripts de conversión ==="
git clone --depth 1 https://github.com/ggerganov/llama.cpp || echo "Ya existe llama.cpp"
cd llama.cpp
pip install -r requirements.txt

echo "=== 2. Descargando modelo desde Hugging Face ==="
mkdir -p ../models/$MODEL_NAME
cd ../models/$MODEL_NAME

# Solo descarga si no existen
[ ! -f config.json ] && huggingface-cli download $HF_REPO --include config.json
[ ! -f tokenizer.model ] && huggingface-cli download $HF_REPO --include tokenizer.model
[ ! -f tokenizer_config.json ] && huggingface-cli download $HF_REPO --include tokenizer_config.json
[ ! -f pytorch_model-00001-of-00002.bin ] && huggingface-cli download $HF_REPO --include pytorch_model-*.bin
[ ! -f generation_config.json ] && huggingface-cli download $HF_REPO --include generation_config.json

cd ../../llama.cpp

echo "=== 3. Ejecutando conversión a GGUF (float16) ==="
python3 convert_hf_to_gguf.py \
  --outfile ../models/$MODEL_NAME/$GGUF_NAME \
  --outtype f16 \
  ../models/$MODEL_NAME

echo "=== 4. Creando archivo Modelfile para Ollama ==="
cd ../models/$MODEL_NAME
cat > Modelfile <<EOF
FROM llama2
PARAMETER temperature 0.7
WEIGHTS $GGUF_NAME
TEMPLATE mistral
EOF

echo "=== 5. Creando modelo Ollama personalizado: $OLLAMA_MODEL_NAME ==="
ollama create $OLLAMA_MODEL_NAME -f Modelfile

echo "=== 6. ¡Listo! Puedes ejecutarlo con: ollama run $OLLAMA_MODEL_NAME ==="
