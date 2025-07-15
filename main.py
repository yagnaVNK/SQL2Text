#!/usr/bin/env python3
# sql_qa_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  


#llm = None
query_cleaner = None
query_generator = None
data_summarizer = None

active_sessions: Dict[str, Dict] = {}

def initialize_llm(model_name="claude-sonnet-4-20250514"):
    """Initialize the LLM model"""
    global  query_cleaner, query_generator, data_summarizer
    
    try:
        # query_cleaner = ChatOllama(model=model_name, temperature=1)
        # logger.info(f"LLM initialized with model: {model_name}")
        # query_generator = ChatOllama(model=model_name, temperature=1)
        # logger.info(f"LLM initialized with model: {model_name}")
        # data_summarizer = ChatOllama(model=model_name, temperature=1)
        # logger.info(f"LLM initialized with model: {model_name}")
        query_cleaner = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=1)
        logger.info(f"LLM initialized with model: {model_name}")
        query_generator = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=1)
        logger.info(f"LLM initialized with model: {model_name}")
        data_summarizer = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=1)
        logger.info(f"LLM initialized with model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        return False

def create_temp_db_uri(base_db_uri, user_id):
    """
    Create a proper temporary database URI for the user
    """

    user_id = user_id.upper()
    
    if base_db_uri.startswith('sqlite:///'):
        db_path = base_db_uri.replace('sqlite:///', '')
        db_name, db_ext = os.path.splitext(db_path)
        temp_db_path = f"{db_name}_{user_id}{db_ext}"
        return f"sqlite:///{temp_db_path}"
    else:
        return f"{base_db_uri}_{user_id}"

def load_data(user_id, db_uri, user_role, chat_log_file):
    """
    This function loads the user Data into the state after the login.
    """
    try:
        user_id = user_id.upper()
        
        df = pd.read_excel(
            "./Files/Student Schedule Dataset with Dimension Description.xlsx", 
            sheet_name="Sheet1"
        )
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # Create proper temporary database URI
        temp_db_URI = create_temp_db_uri(db_uri, user_id)
        logger.info(f"Creating temporary database: {temp_db_URI}")
        
        engine = create_engine(temp_db_URI, echo=False)
        
        if user_role == "student":
            df['student_id'] = df['student_id'].astype(str).str.upper()
            filtered = df[df["student_id"] == user_id]
        elif user_role == "advisor":
            df['advisor_name'] = df['advisor_name'].astype(str).str.upper()
            df['student_id'] = df['student_id'].astype(str).str.upper()
            student_ids = df[df["advisor_name"] == user_id]["student_id"].unique()
            filtered = df[df["student_id"].isin(student_ids)]
        elif user_role == "professor":
            df['instructor_name'] = df['instructor_name'].astype(str).str.upper()
            df['student_id'] = df['student_id'].astype(str).str.upper()
            student_ids = df[df["instructor_name"] == user_id]["student_id"].unique()
            filtered = df[df["student_id"].isin(student_ids)]
        elif user_role == "admin":
            df['student_id'] = df['student_id'].astype(str).str.upper()
            filtered = df.copy()
        else:
            raise ValueError(f"Unknown role: {user_role}")
        
        if filtered.empty and user_role != "admin":
            logger.error(f"No data found for {user_role} {user_id}")
            raise ValueError(f"User '{user_id}' does not exist in the system for role '{user_role}'")
        
        filtered.to_sql("students_details", con=engine, if_exists="replace", index=False)
        logger.info(f"Data loaded successfully for user {user_id} with role {user_role}. Records: {len(filtered)}")
        
        with engine.connect() as conn:
            test_result = conn.execute(text("SELECT COUNT(*) FROM students_details"))
            count = test_result.scalar()
            logger.info(f"Verified {count} records in temporary database for user {user_id}")
        
        return user_id, temp_db_URI, user_role, chat_log_file
        
    except Exception as e:
        logger.error(f"Error loading data for user {user_id}: {str(e)}")
        raise e

def get_chat_history_file(user_id):
    """Get the chat history file path for a user"""

    user_id = user_id.upper()
    os.makedirs("chat_histories", exist_ok=True)
    return f"chat_histories/chat_history_{user_id}.txt"

def load_chat_history(user_id):
    """Load chat history from file"""

    user_id = user_id.upper()
    
    file_path = get_chat_history_file(user_id)
    history = []
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            history.append((entry['question'], entry['sql'], entry['answer']))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid line in {file_path}: {line}")
                            continue
    except Exception as e:
        logger.error(f"Error loading chat history for user {user_id}: {str(e)}")
    
    return history

def save_chat_entry(user_id, question, sql, answer):
    """Save a single chat entry to file"""

    user_id = user_id.upper()
    
    file_path = get_chat_history_file(user_id)
    
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'sql': sql,
                'answer': answer
            }
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        logger.error(f"Error saving chat entry for user {user_id}: {str(e)}")

def clear_chat_history(user_id):
    """Clear chat history file for a user"""

    user_id = user_id.upper()
    
    file_path = get_chat_history_file(user_id)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Chat history cleared for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history for user {user_id}: {str(e)}")
        return False

def generate_sql_query(schema, question, llm, history, user_id, user_role, error=None):
    """
    Use an LLM to generate (or fix) a single SQL query against the given schema,
    taking into account the history of previous Q&A if necessary.
    """

    user_id = user_id.upper()
    
    prev_ctx = ""
    if history:
        prev_q, prev_sql, _ = history[-1]
        prev_ctx = (
            f"Previous question: {prev_q}\n"
            f"Previous SQL: {prev_sql}\n\n"
        )
    err_ctx = f"Last error: {error}\n\n" if error else ""

    role_context = f"""
    Current User: {user_id}
    User Role: {user_role}

    IMPORTANT: You are answering for a {user_role} with ID/Name: {user_id}.
    - If role is 'student': Only show data relevant to this student
    - If role is 'advisor': Only show data for students advised by this advisor
    - If role is 'professor': Only show data for students taught by this professor  
    - If role is 'admin': Can access all data

    The database has been pre-filtered based on the user's role and permissions.
    """
        
    system_message = f"""
    You are an expert assistant that writes SQLite queries. 

    CRITICAL SQL RULES:
    - ALWAYS start SELECT statements with "SELECT DISTINCT" - never use "SELECT" alone
    - Do not add any special characters before and after the SQL query
    - Write clean, properly formatted SQL queries

    {role_context}

    Here is the schema:
    {schema}

    Based on the previous error try to fix the query.
    The previous query is:
    {prev_ctx} 
    and the Error is:
    {err_ctx}

    SQL FORMAT REQUIREMENTS:
    - Every SELECT must be "SELECT DISTINCT" (MANDATORY)
    - Use proper SQLite syntax
    - Include appropriate WHERE clauses based on user role
    - Use table and column names exactly as shown in schema

    If the question is unrelated to the schema, return exactly:
        SELECT DISTINCT NULL;

    EXAMPLES:
    Correct: SELECT DISTINCT student_name FROM students_details WHERE advisor_name = 'John'
    Wrong: SELECT student_name FROM students_details WHERE advisor_name = 'John'

    Correct: SELECT DISTINCT course_code, credits FROM students_details
    Wrong: SELECT course_code, credits FROM students_details

    Respond *only* with the SQL statement (no explanation).
    """.strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    chain = prompt | llm
    ai_msg = chain.invoke({"input": question})
    return ai_msg.content.strip()

def clean_user_question(original_question, llm,schema, user_id, user_role):
    """
    Use LLM to simplify a verbose or natural-language user question
    into a concise and SQL-friendly format for querying.
    """

    user_id = user_id.upper()
    
    role_context = f"""
    You are cleaning questions for a {user_role} with ID/Name: {user_id}.
    Keep in mind the user's role when cleaning the question and preserve role-based relationships.
    """

    data_definition = """
    Student ID	String	Unique identifier for the student
    Student Name	String	Full name of the student
    Advisor Name	String	Advisor assigned to the student
    Course Code	String	Course code (e.g., CS101)
    Course Name	String	Full name of the course
    Term	String	Term in which the course is taken (e.g., Fall 2024)
    Instructor Name	String	Name of the instructor teaching the course
    Days	String	Days of the week the course is held (e.g., MWF, TR)
    Time	String	Time at which the course is scheduled
    Building	String	Building where the course is held
    Room Number	String	Room number in the assigned building
    Credits	String	Number of credits for the course (typically 3 or 4)
    """

    system_prompt = f"""
    You are a system that rewrites user questions for a SQL database assistant.

    {role_context}

    Below is the data definition of available rows:
    {data_definition}

    Below is the schema:
    {schema}

    Your task is to:
    - Preserve role-based relationships (e.g., "my students" for advisors, "my courses" for students)
    - Remove unnecessary conversational context
    - Focus on the core data retrieval intent
    - Use clear and neutral language
    - Maintain the semantic meaning of possessive pronouns based on user role
    - Do not generate any insert, update, or delete queries
    - Keep role-specific filters implicit in the cleaned question

    Role-based relationship rules:
    - If user_role is "advisor": "my students" → "students advised by {user_id}"
    - If user_role is "student": "my courses" → "courses taken by {user_id}"
    - If user_role is "instructor": "my courses" → "courses taught by {user_id}"

    Examples:
    User (advisor): "I am advisor John and what are my students' courses and credits?"
    → Cleaned: "What are the courses and credits for students advised by John?"

    User (student): "Can you show me my enrolled courses this semester?"
    → Cleaned: "What courses is the student enrolled in this semester?"

    User (advisor): "what are my students courses and credits?"
    → Cleaned: "What are the courses and credits for students advised by {user_id}?"

    User (student): "what grades did I get in my courses?"
    → Cleaned: "What grades did the student get in their courses?"

    User (admin): "show me all courses for student SID04"
    → Cleaned: "What courses is student SID04 taking?"

    If the question is already clean and unambiguous, return it unchanged.
    Only return the cleaned question and nothing else.
    """.strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    cleaned = chain.invoke({"input": original_question})
    return cleaned.content.strip()

def generate_final_answer(question, history, new_details, llm, user_id, user_role):
    """
    Use LLM to simplify a detailed SQL table or dataframe into a simple concise reply in a conversational way.
    """

    user_id = user_id.upper()
    
    role_context = f"""
    You are answering as a helpful assistant for a {user_role} with ID/Name: {user_id}.
    Tailor your response appropriately for their role and context.
    """
            
    prev_ctx = ""
    prev_answer = ""
    if history:
        prev_q, prev_sql, prev_answer = history[-1]
        prev_ctx = (
            f"Previous question: {prev_q}\n"
            f"Previous SQL: {prev_sql}\n\n"
        )

    data_definition = """
    Student ID: Unique identifier for the student
    Student Name: Full name of the student  
    Advisor Name: Advisor assigned to the student
    Course Code: Course code (e.g., CS101)
    Course Name: Full name of the course
    Term: Term in which the course is taken (e.g., Fall 2024)
    Instructor Name: Name of the instructor teaching the course
    Days: Days of the week the course is held (e.g., MWF, TR)
    Time: Time at which the course is scheduled
    Building: Building where the course is held
    Room Number: Room number in the assigned building
    Credits: Number of credits for the course (typically 3 or 4)
    """

    system_message = f"""
    You are a helpful assistant who answers questions based on database query results.

    {role_context}

    INSTRUCTIONS:
    1. Analyze the data retrieved from the database query below
    2. If the query returned rows of data, interpret and present this information to answer the user's question
    3. If the query returned zero rows (completely empty result set), state "No matching records found"
    4. Answer the user's question directly using the retrieved data

    Data Retrieved from Database:
    {new_details}

    Data Schema Reference:
    {data_definition}

    Previous Context (for reference):
    {prev_ctx}

    Response Guidelines:
    - **For small result sets (≤5 rows)**: Provide a brief summary with key insights
    - **For large result sets (>5 rows)**: Use format "Below are the details requested:" followed by the data
    - **IMPORTANT**: Use plain text format only - NO tables, NO markdown formatting, NO bullet points
    - Present data as simple sentences or comma-separated lists
    - If some fields show NULL/None/empty values, you can mention "not specified" or "not available"
    - Keep responses focused and relevant to the user's question
    - If the result set is empty (no rows returned), then say "No matching records found"

    Response Format Examples:
    - **Small dataset**: "You have 3 courses enrolled: CSC103 (3 credits), CSC104 (4 credits), CSC105 (3 credits)"
    - **Large dataset**: "Below are the details requested: CSC103 (3 credits), CSC104 (4 credits), CSC105 (3 credits), CSC108 (4 credits), CSC109 (3 credits), CSC102 (4 credits)"
    - **Empty result**: "No matching records found for your query"
    - **Summary when appropriate**: "You have 15 courses total with an average of 3.2 credits per course"

    Now answer the user's question based on the data retrieved above.
    """.strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    chain = prompt | llm
    ai_msg = chain.invoke({"input": question})

    return ai_msg.content.strip()

def format_results(rows, columns):
    """
    Format query results into a string representation.
    """
    if not rows:
        return "No results found."
    else:
        df = pd.DataFrame(rows, columns=columns)
        return "\n" + df.to_string(index=False)

def process_question(user_id: str, question: str):
    """
    Process a question for a specific user, maintaining their conversation history.
    """
    global query_cleaner, query_generator, data_summarizer, active_sessions
    
    user_id = user_id.upper()
    
    if not query_cleaner:
        return {
            "success": False,
            "error": "LLM not initialized",
            "answer": None,
            "sql_query": None,
            "data": None
        }
    
    if user_id not in active_sessions:
        return {
            "success": False,
            "error": "User not logged in. Please login first.",
            "answer": None,
            "sql_query": None,
            "data": None
        }
    
    session = active_sessions[user_id]
    temp_db_uri = session['db_uri']  
    user_role = session['user_role']
    
    logger.info(f"Processing question for user {user_id} using database: {temp_db_uri}")
    
    try:
        engine = create_engine(temp_db_uri, echo=False)
        db = SQLDatabase.from_uri(temp_db_uri, sample_rows_in_table_info=3)
        schema = db.table_info
        
        history = load_chat_history(user_id)
        
        cleaned_question = clean_user_question(question, query_cleaner,schema, user_id, user_role)
        
        retries = 10
        last_error = None
        sql = None
        answer = ""
        data = None
        
        while retries:
            sql = generate_sql_query(schema, cleaned_question, query_generator, history, user_id, user_role, last_error)
            
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = result.fetchall()
                    cols = result.keys()
                
                details = format_results(rows, cols)
                
                data = {
                    "columns": list(cols),
                    "rows": [list(row) for row in rows]
                }

                answer = generate_final_answer(cleaned_question, history, details, data_summarizer, user_id, user_role)

                save_chat_entry(user_id, question, sql, answer)
                
                return {
                    "success": True,
                    "error": None,
                    "answer": answer,
                    "sql_query": sql,
                    "data": data,
                    "original_question": question,
                    "cleaned_question": cleaned_question,
                    "user_role": user_role,
                    "database_uri": temp_db_uri 
                }
                
            except Exception as e:
                last_error = str(e)
                retries -= 1
                logger.warning(f"Query failed for user {user_id}: {last_error}, retries left: {retries}")
                
                if retries == 0:
                    return {
                        "success": False,
                        "error": f"Query failed after multiple retries: {last_error}",
                        "answer": "Unable to process your question after multiple attempts.",
                        "sql_query": sql,
                        "data": None
                    }
    
    except Exception as e:
        logger.error(f"Error processing question for user {user_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "answer": "An error occurred while processing your question.",
            "sql_query": None,
            "data": None
        }

def cleanup_temp_database(user_id, db_uri):
    """
    Clean up temporary database file when user logs out
    """
    user_id = user_id.upper()
    
    try:
        if db_uri.startswith('sqlite:///'):
            db_path = db_uri.replace('sqlite:///', '')
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"Cleaned up temporary database: {db_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary database for user {user_id}: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "SQL Q&A API is running"
    })

@app.route('/login', methods=['POST'])
def login():
    """
    Login endpoint to authenticate user and load their data
    Expected JSON payload:
    {
        "user_id": "string",
        "user_role": "student|advisor|professor|admin",
        "db_uri": "optional_db_uri"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        user_id = data.get('user_id')
        user_role = data.get('user_role')
        db_uri = data.get('db_uri', 'sqlite:///student_details.db')
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400
        
        user_id = user_id.upper()
        
        if not user_role:
            return jsonify({
                "success": False,
                "error": "user_role is required"
            }), 400
        
        if user_role not in ['student', 'advisor', 'professor', 'admin']:
            return jsonify({
                "success": False,
                "error": "user_role must be one of: student, advisor, professor, admin"
            }), 400

        chat_log_file = get_chat_history_file(user_id)
        
        try:

            loaded_user_id, temp_db_uri, loaded_role, chat_file = load_data(
                user_id, db_uri, user_role, chat_log_file
            )
            
            active_sessions[user_id] = {
                'user_id': loaded_user_id,
                'user_role': loaded_role,
                'db_uri': temp_db_uri,  
                'original_db_uri': db_uri,  
                'chat_log_file': chat_file,
                'login_time': datetime.now().isoformat()
            }

            history = load_chat_history(user_id)
            
            return jsonify({
                "success": True,
                "message": f"Login successful for {user_role} {user_id}",
                "user_id": user_id,
                "user_role": user_role,
                "temp_database": temp_db_uri,  
                "chat_history_count": len(history)
            })
            
        except Exception as e:
            logger.error(f"Error during login for user {user_id}: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Login failed: {str(e)}"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/logout', methods=['POST'])
def logout():
    """
    Logout endpoint
    Expected JSON payload:
    {
        "user_id": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400
        
        user_id = user_id.upper()
        
        if user_id in active_sessions:
            session = active_sessions[user_id]
            temp_db_uri = session.get('db_uri')

            if temp_db_uri:
                cleanup_temp_database(user_id, temp_db_uri)
            
            del active_sessions[user_id]
            message = f"Logout successful for user {user_id}"
        else:
            message = f"User {user_id} was not logged in"
        
        return jsonify({
            "success": True,
            "message": message
        })
        
    except Exception as e:
        logger.error(f"Error in logout endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Main endpoint for asking questions
    Expected JSON payload:
    {
        "user_id": "string",
        "question": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        user_id = data.get('user_id')
        question = data.get('question')
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400

        user_id = user_id.upper()
        
        if not question:
            return jsonify({
                "success": False,
                "error": "question is required"
            }), 400

        result = process_question(user_id, question)
        
        status_code = 200 if result["success"] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in ask_question endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    """Get conversation history for a specific user from file"""
    try:
        user_id = user_id.upper()
        
        history = load_chat_history(user_id)
        
        formatted_history = []
        for question, sql, answer in history:
            formatted_history.append({
                "question": question,
                "sql_query": sql,
                "answer": answer
            })
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "history": formatted_history,
            "total_conversations": len(formatted_history),
            "is_logged_in": user_id in active_sessions
        })
        
    except Exception as e:
        logger.error(f"Error getting history for user {user_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/clear_history/<user_id>', methods=['DELETE'])
def clear_user_history(user_id):
    """Clear conversation history for a specific user"""
    try:
        user_id = user_id.upper()
        
        success = clear_chat_history(user_id)
        
        if success:
            message = f"History cleared for user {user_id}"
        else:
            message = f"Failed to clear history for user {user_id}"
        
        return jsonify({
            "success": success,
            "message": message
        })
        
    except Exception as e:
        logger.error(f"Error clearing history for user {user_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/active_sessions', methods=['GET'])
def get_active_sessions():
    """Get list of users with active sessions"""
    try:
        sessions_info = []
        for user_id, session_data in active_sessions.items():
            sessions_info.append({
                "user_id": user_id,
                "user_role": session_data['user_role'],
                "login_time": session_data['login_time'],
                "temp_database": session_data['db_uri']
            })
        
        return jsonify({
            "success": True,
            "active_sessions": sessions_info,
            "total_sessions": len(sessions_info)
        })
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":

    if not initialize_llm():
        logger.error("Failed to initialize LLM. Exiting.")
        sys.exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=5000)