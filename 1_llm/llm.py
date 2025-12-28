from langchain_openai import OpenAI
from dotenv import load_dotenv  

load_dotenv()  # Load environment variables from .env file

def get_llm(model_name="gpt-4", temperature=0.7, max_tokens=1500): #temperature  is kind of creativity level we want as our output
    """
    Initialize and return an OpenAI LLM instance with specified parameters.

    Args:
        model_name (str): The name of the model to use (default is "gpt-4").
        temperature (float): The temperature setting for the model (default is 0.7).
        max_tokens (int): The maximum number of tokens to generate (default is 1500).   
        """
    llm = OpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm
# Example usage:
llm_instance = get_llm()  
print(llm_instance("Hello, how are you?"))
