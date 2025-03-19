from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {Question}
"""

model = OllamaLLM(model="llama3.2:latest")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the AI Chatbot! (Type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate response with conversation history
        result = chain.invoke({"context": context, "Question": user_input})
        
        print(f"AI: {result}")

        # Update conversation history
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()
