import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Kimi model hosted on Hugging Face
model_id = "moonshotai/Kimi-K2-Instruct"
provider = "novita"

kimi_llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        provider=provider,
        temperature=0.8,
        max_new_tokens=512,
    )
)

query = input("Please enter your query: ")
response = kimi_llm.invoke(query)
print(response.content)
