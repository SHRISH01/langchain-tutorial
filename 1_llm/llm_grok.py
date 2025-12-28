import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "reasoning": {"enabled": True}
        }
    }
)

response = llm.invoke("write bad things about name simran .")
print(response.content)
