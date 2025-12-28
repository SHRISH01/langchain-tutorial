from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0) # temperature set to 0 for deterministic output
result = model.invoke("Hello! How are you?")
print(result)
print(result.content)