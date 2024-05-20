# Prompts and LLMs
import os

os.environ["OPENAI_API_KEY"] = "..."

from langchain import PromptTemplate #Modular way to create a prompt and replace with variable values 
from langchain.llms import OpenAI
from langchain.chains import LLMChain #take template and chain it with LLM, LLMChain are simplest chain in LangChain

template = "You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?"

prompt = PromptTemplate.from_template(template)
llm = OpenAI(temperature=0.9)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run({"company": "ABC Startup", "product": "colorful socks"}))

#template = "How to deal with {problem}"
#prompt=PromptTemplate.from_template(template)
#print(prompt.format(problem="anxiety"))
#llm= HuggingFaceHub(repo_id="google/flan-t5-base",model_kwargs={"temperature":1, "max_length":64})
#chain=LLMChain(llm=llm,prompt=prompt)
#print(chain.run({"problem":"lazy"}))
