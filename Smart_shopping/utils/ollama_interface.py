import requests
import json

def ollama_llm(prompt, model="llama2:7b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    
    # Process the response as a stream of JSON objects
    try:
        responses = []
        for line in response.text.splitlines():
            try:
                json_obj = json.loads(line)  # Parse each line as JSON
                if "response" in json_obj:
                    responses.append(json_obj["response"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}, Error: {e}")
        
        # Combine all the 'response' parts into a single string
        return "".join(responses)
    except Exception as e:
        print("Error processing response:", e)
        return "Error: Unable to process the response from the LLM."