FROM ollama/ollama:latest

# Copy the entrypoint file
COPY entrypoint.sh /entrypoint.sh

# Set the permission
RUN chmod +x /entrypoint.sh

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]