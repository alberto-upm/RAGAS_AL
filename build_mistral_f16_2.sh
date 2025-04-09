#!/bin/bash

set -e  # Termina si hay error

# Definición de rutas y nombres
HF_REPO="mistralai/Mistral-7B-Instruct-v0.1"
MODEL_NAME="mistral-7b-instruct"
GGUF_NAME="mistral-f16.gguf"
OLLAMA_MODEL_NAME="mistral-f16"

# Ruta absoluta base (ajusta si estás en otro lugar)
BASE_DIR="$(pwd)"
MODEL_DIR="$BASE_DIR/models/$MODEL_NAME"

: << 'EOF'

echo "=== 0. Clonando llama.cpp y descargando scripts de conversión ==="
git clone --depth 1 https://github.com/ggerganov/llama.cpp || echo "Ya existe llama.cpp"
cd llama.cpp
pip install -r requirements.txt

echo "=== 1. Descargando modelo desde Hugging Face (requiere login) ==="
mkdir -p $MODEL_DIR
cd $MODEL_DIR

# Solo descarga si no existen
huggingface-cli download $HF_REPO --include config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download $HF_REPO --include tokenizer.model --local-dir . --local-dir-use-symlinks False
huggingface-cli download $HF_REPO --include tokenizer_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download $HF_REPO --include generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download $HF_REPO --include pytorch_model-00001-of-00002.bin --local-dir . --local-dir-use-symlinks False
huggingface-cli download $HF_REPO --include pytorch_model-00002-of-00002.bin --local-dir . --local-dir-use-symlinks False


cd $BASE_DIR

echo "=== 2. Convirtiendo a GGUF en float16 usando convert_hf_to_gguf.py ==="
python3 $BASE_DIR/llama.cpp/convert_hf_to_gguf.py \
    models/$MODEL_NAME \
    --outfile models/$MODEL_NAME/$GGUF_NAME \
    --outtype f16
EOF
echo "=== 3. Creando Modelfile para Ollama ==="
cd models/$MODEL_NAME
cat > Modelfile <<EOF
FROM mistral
TEMPLATE mistral
EOF

echo "=== 4. Creando modelo en Ollama: $OLLAMA_MODEL_NAME ==="
ollama create $OLLAMA_MODEL_NAME -f Modelfile

echo "=== 5. ¡Listo! Ejecuta tu modelo con: ollama run $OLLAMA_MODEL_NAME ==="
