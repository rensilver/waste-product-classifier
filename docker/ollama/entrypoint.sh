#!/bin/sh
set -e

ollama serve &
SERVER_PID=$!

echo "Waiting for the Ollama server to come up..."
until ollama list >/dev/null 2>&1; do
    sleep 1
done

echo "Pulling model: ${OLLAMA_MODEL}"
ollama pull "${OLLAMA_MODEL}"
echo "Model ${OLLAMA_MODEL} ready."

wait "${SERVER_PID}"
