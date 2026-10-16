import ollama

import config

# Smoke test: confirms Ollama is running and the configured model (OLLAMA_MODEL, default llama3) answers.
model_name = config.LLM_MODEL
# Define the user input prompt
prompt = "Explain what Agent AI is in simple words."
# Run the model and get a response
response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
# Print the response
print("AI Response:", response["message"]["content"])
