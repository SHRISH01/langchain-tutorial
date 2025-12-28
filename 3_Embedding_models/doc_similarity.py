from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-large-3",dimensions=1536)
document = ["LangChain is a framework for developing applications powered by language models.",
            "It enables the creation of chatbots, Generative Question-Answering (GQA) systems, and summarization tools.",
            "LangChain provides modules for prompt management, memory, and integrations with various data sources.",
            "The framework supports multiple language models and offers utilities for building complex applications easily.",
            "LangChain is widely used for developing AI-driven applications that leverage natural language processing capabilities.",
            "It has a strong community and extensive documentation to help developers get started quickly.",
            "LangChain's modular design allows for easy customization and extension of its functionalities.",
            "The framework is designed to facilitate rapid prototyping and deployment of language model applications.",
            "LangChain supports both cloud-based and on-premises deployment options for flexibility.",
            "It is an open-source project that encourages contributions from developers around the world."
            ]

query = "What is LangChain used for?"

doc_emb = embedding.embed_documents(document)
query_emb = embedding.embed_query(query)
similarity_scores = cosine_similarity([query_emb], doc_emb)[0]

print("Similarity Scores:")
for i, score in enumerate(similarity_scores):
    print(f"Document {i+1}: {score:.4f}")