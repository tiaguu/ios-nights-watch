FROM ollama/ollama:latest

EXPOSE 11434

RUN chmod +x ollama.sh

# Run .sh file
CMD ["ollama.sh"]