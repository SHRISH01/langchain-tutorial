from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic()

prompt1 = PromptTemplate(
    template="Generate short notes on the follqwing text: {text}",
    input_variables=["text"]
)
prompt2 = PromptTemplate(
    template="Generate short questions based on the following text: {text} ",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the following notes and short questions into one , notes : {notes} and questions : {questions}",
    input_variables=["notes", "questions"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': model1 | prompt1 | parser,
    'questions': model2 | prompt2 | parser
})

merge_chain = parallel_chain | prompt3 | parser

result = merge_chain.invoke({
    'text': "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These intelligent machines can perform tasks such as problem-solving, decision-making, language understanding, and visual perception. AI can be categorized into narrow AI, which is designed for specific tasks, and general AI, which possesses the ability to perform any intellectual task that a human can do. The field of AI encompasses various subfields, including machine learning, natural language processing, computer vision, and robotics. Advances in AI have led to significant developments in various industries, including healthcare, finance, transportation, and entertainment."
})
print(result)

# For visualization pof the chain

merge_chain.visualize().render("parallel_chain_graph")
