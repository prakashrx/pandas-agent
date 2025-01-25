import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SimpleAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="dummy-api-key",
        )
        self.system_message = """You are a helpful assistant. 
            Keep responses concise and informative."""
        
    def generate_response(self, user_input):
        response = self.client.chat.completions.create(
            model="local-model",  # Model name should match what's in LM Studio
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    agent = SimpleAgent()
    print("Agent initialized. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = agent.generate_response(user_input)
        print(f"Agent: {response}")