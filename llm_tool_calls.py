from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

llm = ChatOpenAI(model ="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content = response.content))

    return state

graph = StateGraph(AgentState)

graph.add_node("chatModel",process)
graph.add_edge(START,"chatModel")
graph.add_edge("chatModel",END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    print(f"AI: {conversation_history[-1].content}")
    user_input = input("Enter: ")

