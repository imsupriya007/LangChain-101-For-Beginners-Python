# Simple Sequential Chains
import os

os.environ["OPENAI_API_KEY"] = "..."

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import SimpleSequentialChain #Output of one chain is input to another

llm = OpenAI(temperature=0)
#llm= HuggingFaceHub(repo_id="google/flan-t5-base",model_kwargs={"temperature":1, "max_length":64})
template = "What is a good name for a company that makes {product}?"
first_prompt = PromptTemplate.from_template(template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

second_template = "Write a catch phrase for the following company: {company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True) #how chain performs at each call - verbose monitors
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)
