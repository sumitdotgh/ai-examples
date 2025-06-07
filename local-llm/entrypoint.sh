#!/bin/sh
ollama serve &       # start the server in background
sleep 5              # wait for server to be ready
ollama pull mistral  # pull the model now that server is running
wait                 # keep container running
