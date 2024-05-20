# Action Agents
# - flexiblity to chain a series of calls and other tools based on the user input to get to an answer
# - access to suit of tools and determine step by step which one to use depending on what the user prompts
import os

os.environ["OPENAI_API_KEY"] = "..."

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent

prompt = "When was the 3rd president of the United States born? What is that year raised to the power of 3?"
# the answer fetched by LLM will be wrong as weak in math calculation and current affairs
# use tools = action Agents take, like use wikipedia or llm-maths
#tools used depend on the prompt

import pprint
from langchain.agents import get_all_tool_names
pp= pprint.PrettyPrinter(indent=4)
pp.pprint(get_all_tool_names())  #list all tools available for Agents

llm = OpenAI(temperature=0)
#llm = HuggingFaceHub(repo_id="google/flan-t5-base",model_kwargs={"temperature":0.5, "max_length":64})
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True) #how chain performs at each call - verbose monitors
# A zero shot agent - uses React framework to determine which tool to use based on tool's description. That does a reasoning step before acting

agent.run(prompt)
