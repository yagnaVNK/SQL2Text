#!/usr/bin/env python3
# sql_qa.py

import sys
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

id = "SID01"

def generate_sql_query(schema, question, llm, history, error=None):
    """
    Use an LLM to generate (or fix) a single SQL query against the given schema,
    taking into account the history of previous Q&A if necessary.
    """
    # Build a little “previous context” string:
    prev_ctx = ""
    if history:
        prev_q, prev_sql,_ = history[-1]
        prev_ctx = (
            f"Previous question: {prev_q}\n"
            f"Previous SQL: {prev_sql}\n\n"
        )
    err_ctx = f"Last error: {error}\n\n" if error else ""
    system_message = f"""
You are an expert assistant that writes SQLite queries. You always add distinct keyword for all the selected columns and do not add any special characters before and after sql query.
Here is the schema:
{schema}
Based on the previous error try to fix the query.
The previous query is
{prev_ctx} 
and the Error is 
{err_ctx}
If the question is unrelated to the schema, return exactly:

    SELECT NULL;

Respond *only* with the SQL statement (no explanation). 
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    chain = prompt | llm
    ai_msg = chain.invoke({"input": question})
    return ai_msg.content.strip()

def clean_user_question(original_question, llm):
    """
    Use LLM to simplify a verbose or natural-language user question
    into a concise and SQL-friendly format for querying.
    """
    system_prompt = """
    You are a system that rewrites user questions for a SQL database assistant.

    Your task is to:
    - Remove unnecessary context (like "I am professor", "please", "can you", etc.)
    - Focus on the core data retrieval intent
    - Use clear and neutral language
    - Include only entities relevant to the query (e.g., student names, IDs, advisor info)
    - Avoid any role-based or narrative context

    Examples:
    User: I am professor 1 and which advisor should I contact regarding student 2?
    → Cleaned: I want to know the advisor of student 2.

    User: Can you tell me what department student John Smith belongs to?
    → Cleaned: What department is student John Smith in?

    User: As an admin, I want to find out all grades for student 123.
    → Cleaned: What are the grades of student 123?

    If the question is already clean, return it unchanged.
    Only return the cleaned question and nothing else.
    """.strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    cleaned = chain.invoke({"input": original_question})
    return cleaned.content.strip()


def generate_final_answer(question, history, new_details, llm):
    prev_ctx = ""
    prev_answer = ""
    if history:
        prev_q, prev_sql, prev_answer = history[-1]
        prev_ctx = (
            f"Previous question: {prev_q}\n"
            f"Previous SQL: {prev_sql}\n\n"
        )

    system_message = f"""
    You are a helpful assistant who answers questions using the data retrieved from a database. The results are provided in text format.

    Below is the data retrieved from the database:

    {new_details}

    Your job is to:
    - Use all unique rows and values from this data to craft a natural and complete answer.
    - Mention every unique detail found in the data — no summarizing or skipping over data.
    - Phrase the answer clearly and conversationally, like you're explaining to someone unfamiliar with databases.
    - Do not hallucinate or guess anything not in the data.
    - If there are NULL, None, or missing values, say: "No information available" for those parts.
    - If the result is empty, reply: "No matching records found or access denied."

    Always stay grounded in the actual data provided.
    """.strip()


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    chain = prompt | llm
    ai_msg = chain.invoke({"input": question})

    return ai_msg.content.strip()


def format_and_print(rows, columns):
    """
    Nicely display query results.
    """
    details = ""
    if not rows:
        details = "No results found."
    else:
        df = pd.DataFrame(rows, columns=columns)
        details = "\n" + df.to_string(index=False)
    print(details)
    return details

def main(db_uri="sqlite:///student_details.db", model_name="llama3.2"):
    # 1) Connect
    engine = create_engine(db_uri, echo=False)
    db      = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
    schema  = db.table_info  # list of table-info strings
 
    # 2) LLM for SQL generation
    llm = ChatOllama(model=model_name, temperature=1)

    print(f" Connected to database: {db_uri}")

    history = []  
    while True:
        question = input("❓  Enter your question (or 'exit' to quit): ").strip()
        if question.lower() in ("exit", "quit"):
            print(" Goodbye!")
            sys.exit(0)
        question = clean_user_question(question,llm)
        retries    = 10
        last_error = None
        sql        = None
        answer = ""
        while retries:
            sql = generate_sql_query(schema, question, llm, history, last_error)
            print("\nGenerated SQL:")
            print(sql)
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows    = result.fetchall()
                    cols    = result.keys()
                # store in history before formatting output
                
                details = format_and_print(rows, cols)
                answer = generate_final_answer(question,history,details,llm)
                history.append((question, sql, answer))
                break
            except Exception as e:
                last_error = str(e)
                print(f"\nQuery failed: {last_error}")
                retries -= 1
                if retries:
                    print(f"   Retrying ({retries} attempts left)…\n")
                else:
                    print("   Giving up after multiple retries.\n")
                    answer = "Giving up after multiple retries."
        print(answer)
            
if __name__ == "__main__":
    main()
