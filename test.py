from langchain.chains import LLMChain
from langchain.llms import OpenAIfrom 
from langchain.prompts import PromptTemplate
from langchain.chains import SimplesSequentialChain

template = """당신은 발달장애 아동과 소통을 하는 챗봇입니다. 발달장애 아동이 질문을 했을 때 친구처럼 
답변해 주는 것이 당신의 임무입니다. """

질문:{question}


prompt = PromptTemplate(
    input_variables=["question"],
    template = template
)

overall_chain = SimplesSequentialChain(
    chains=[chain1, chain2],
    verbose = True
)