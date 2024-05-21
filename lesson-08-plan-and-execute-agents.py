# Plan and Execute Agents
# first plan series of steps and then execute each one
# LLM is not good in current affairs, wikipedia ques, math ques
# Ex: Prompt - where is the next summer olympics going to be held? whats population of the country raised to power 0.43
# Steps are 1. Search for the location 2. Identify the country 3. search for country population 4. Raise population to power 0.43 5. Return the result

import os

os.environ["OPENAI_API_KEY"] = "..."

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain import SerpAPIWrapper, LLMMathChain
from langchain.agents.tools import Tool


search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(
    "Where will the next summer olympics be hosted? What is the population of that country raised to the 0.43 power?"
)
