from langchain.llms import Ollama
from langchain.utilities import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt


# 1. Connect to PostgreSQL database
engine = create_engine("postgresql+psycopg2://postgres:nrm001@localhost:5432/inventory_db")
db = SQLDatabase(engine)

# 2. SQL capture
captured_sql = {"query": ""}

def run_sql_tool(sql_query: str) -> str:
    captured_sql["query"] = sql_query
    try:
        return db.run(sql_query)
    except Exception as e:
        return f"Error: {e}"

# 3. SQL tool
tools = [
    Tool(
        name="sql_query",
        func=run_sql_tool,
        description="Use this tool to run SQL queries on the PostgreSQL database. Input must be a complete SQL SELECT query."
    )
]

# 4. LLM with system prompt
llm = Ollama(
    model="llama3.2",
    system="""
You are a data analysis assistant connected to a PostgreSQL database.

- Always use the `sql_query` tool to retrieve data before answering.
- The main data is in a table named `product`, unless the user says otherwise.
- If the question suggests a chart or summary, generate a SQL SELECT query and call the tool.
- Only return data after executing the SQL query with the tool.

Respond like this:

Action: sql_query
Action Input: SELECT category, AVG(price) FROM product GROUP BY category;
"""
)

# 5. Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 6. Interactive question loop
print("\n🔁 Ask questions (type 'exit' to quit):")
while True:
    question = input("\n❓ Your Question: ")
    if question.strip().lower() in {"exit", "quit"}:
        print("👋 Exiting.")
        break

    # Run agent
    try:
        agent.run(question)

        # Extract SQL and get DataFrame
        sql = captured_sql["query"]
        if sql.strip().lower().startswith("select"):
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)

            print("\n📊 Query Result:")
            print(df)

            # Plot if numerical
            if len(df.columns) >= 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
                df.plot(kind="bar", x=df.columns[0], y=df.columns[1], legend=False)
                plt.title(question)
                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")