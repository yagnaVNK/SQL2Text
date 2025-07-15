import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from langchain.schema import AgentFinish
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq


df = pd.read_excel("./Files/Student Schedule Dataset with Dimension Description.xlsx", sheet_name="Sheet1")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

connection = sqlite3.connect("student_details.db")
df.to_sql(name="students", if_exists="replace",con=connection)

@dataclass
class Event:
    event: str
    timestamp: str
    text: str


def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        assert len(prompts) == 1
        event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        generation = response.generations[-1][-1].message.content
        event = Event(event="llm_end", timestamp=_current_time(), text=generation)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")



my_llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434")
llm = ChatOllama(model ="llama3.2")
db = SQLDatabase.from_uri("sqlite:///student_details.db")

@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    print("Tool call----> list tables tool called")
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    print("Tool call----> tables_schema tool called")
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    print("Tool call----> execute_sql tool called")
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    print("Tool call----> check_sql tool called")
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

sql_dev = Agent(
    role="Senior Database Developer",
    goal="Construct and execute SQL queries based on a request",
    backstory=dedent(
        """
        You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `execute_sql` to check your queries for correctness.
        Use the `check_sql` to execute queries against the database.
    """
    ),
    verbose=True,
    llm=my_llm,
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=False,
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal="You receive data from the database developer and analyze it",
    backstory=dedent(
        """
        You have deep experience with analyzing datasets using Python.
        Your work is always based on the provided data and is clear,
        easy-to-understand and to the point. You have attention
        to detail and always produce very detailed work (as long as you need).
    """
    ),
    verbose=True,
    llm=my_llm,
    allow_delegation=False,
)

report_writer = Agent(
    role="Senior Report Editor",
    goal="Write an executive summary type of report based on the work of the analyst",
    backstory=dedent(
        """
        Your writing still is well known for clear and effective communication.
        You always summarize long texts into bullet points that contain the most
        important details.
        """
    ),
    verbose=True,
    llm=my_llm,
    allow_delegation=False,
)

extract_data = Task(
    description="Extract data that is required for the query {query}.",
    expected_output="Database result for the query",
    agent=sql_dev,
)

analyze_data = Task(
    description="Analyze the data from the database and write an analysis for {query}.",
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data],
)


write_report = Task(
    description=dedent(
        """
        Write an executive summary of the report from the analysis. The report
        must be less than 100 words.
    """
    ),
    expected_output="Markdown report",
    agent=report_writer,
    context=[analyze_data],
)

crew = Crew(
    agents=[sql_dev, data_analyst, report_writer],
    tasks=[extract_data, analyze_data, write_report],
    process=Process.sequential,
    verbose=True,
    memory=False,
    output_log_file="crew.log",
)

inputs = {
    "query": "What are the courses of Student 01?"
}

result = crew.kickoff(inputs=inputs)
print(result)