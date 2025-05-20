import time
import psutil
import psycopg2
import ollama
from typing import List, Dict


# ---------------- CONFIG ----------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "inventory_db",
    "user": "postgres",
    "password": "nrm001"
}

models_to_test = ['llama3.2', 'llama3:instruct','codellama:7b','codellama:13b','deepseek-r1:32b','deepseek-r1:14b','deepseek-r1:8b','gemma']  

def get_table_list() -> List[str]:
    """Query DB for list of tables."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    return tables


def query_model(model_name: str, table_list: List[str]) -> Dict:
    """Use Ollama model to interpret database schema."""
    prompt = f"Given the following tables in a PostgreSQL database:\n\n{', '.join(table_list)}\n\nDescribe what this database might be used for."

    start_time = time.time()

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        elapsed = time.time() - start_time
        

        return {
            "model": model_name,
            "time_sec": elapsed,
            "response": response['message']['content']
        }
    except Exception as e:
        return {
            "model": model_name,
            "time_sec": -1,
            "response": f"Error: {e}"
        }


def compare_models():
    print(" Connecting to database...")
    tables = get_table_list()
    print(f"✅ Found {len(tables)} tables: {tables}\n")

    results = []
    for model in models_to_test:
        print(f"Testing model: {model}")
        result = query_model(model, tables)
        results.append(result)

    print("\n--- Results ---\n")
    for res in results:
        print(f"Model: {res['model']}")
        print(f"Time taken: {res['time_sec']:.3f} sec")
        print(f"Response: {res['response'][:250]}...\n")

if __name__ == "__main__":
    compare_models()
