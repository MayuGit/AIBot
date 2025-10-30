import gradio as gr
from ollama import Client

# --- Configuration ---
# On Docker Desktop (Windows/Mac), 'host.docker.internal' is used 
# to access the host's services (like the Ollama container)
OLLAMA_BASE_URL = 'http://host.docker.internal:11434'
MODEL_NAME = 'default'

# Initialize the Ollama Client
client = Client(host=OLLAMA_BASE_URL)

def chat_with_gemma(message, history):
    # Gradio history format needs to be converted to Ollama's 'messages' format
    messages = []

    # Add past conversation history
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})

    # Add the current user message
    messages.append({"role": "user", "content": message})

    # Call the Ollama API with streaming enabled
    stream = client.chat(
        model=MODEL_NAME, 
        messages=messages,
        stream=True
    )

    full_response = ""
    # Stream the response back to Gradio
    for chunk in stream:
        if 'content' in chunk['message']:
            # Yield the response chunk by chunk for real-time display
            full_response += chunk['message']['content']
            yield full_response

# --- Build the Gradio Interface ---
gr.ChatInterface(
    fn=chat_with_gemma,
    textbox=gr.Textbox(placeholder=f"Ask {MODEL_NAME} a question...", container=False, scale=7),
    title=f"Local Gemma 3n Chatbot (via Ollama + GPU)",
    theme="soft",
    examples=[
        "Explain the difference between a container and a VM.", 
        "Write a short, pirate-themed haiku."
    ],
    #retry_btn="Try Again",
    #undo_btn="Delete Last",
    #clear_btn="Clear History"
).launch(server_name="0.0.0.0", server_port=7860) 
# '0.0.0.0' allows external access (from Docker)