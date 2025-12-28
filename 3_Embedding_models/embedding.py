from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",dimensions=32)

result = embeddings.aembed_query(embeddings)
print(str(result))

# For Documents we can use 
# embeddings.embed_documents()