from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()   

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-2-7b-chat-hf', task='text-generation', model_kwargs={'temperature': 0, 'max_new_tokens': 500})

model = ChatHuggingFace(llm=llm)
response = model.invoke("Capital Of India?")
print(response)