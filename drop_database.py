import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import re

# Your connection details
USER = 'root'
PASSWORD = '12345'
HOST = '127.0.0.1'  # Make sure your host is correct (it should match your MySQL server's address)
PORT = '3306'  # Default MySQL port

# Location of your .env file
ENV_FILE_PATH = '.env'

# New database name
NEW_DB_NAME = 'heartpred'
OLD_DB_NAME = 'exchange2'

# Create a connection URL without specifying a database (so we can connect to the MySQL server)
DATABASE_URI = f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}'

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URI)

# Function to remove the old database and create the new one
def update_database():
    try:
        # Connect to MySQL server (without specifying a database)
        with engine.connect() as conn:
            # Drop the 'exchange2' database if it exists
            conn.execute(text(f"DROP DATABASE IF EXISTS {OLD_DB_NAME}"))
            print(f"Database '{OLD_DB_NAME}' dropped successfully.")
            
            # Create the 'heartpred' database
            conn.execute(text(f"CREATE DATABASE {NEW_DB_NAME}"))
            print(f"Database '{NEW_DB_NAME}' created successfully.")
        
        # Now, update the .env file to point to the new database
        update_env_file()

    except OperationalError as e:
        print(f"An error occurred: {e}")

# Function to update the .env file
def update_env_file():
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as file:
            env_content = file.read()

        # Replace the old database name 'exchange2' with 'heartpred'
        new_env_content = re.sub(r'(\bexchange2\b)', NEW_DB_NAME, env_content)

        # Write the updated content back to the .env file
        with open(ENV_FILE_PATH, 'w') as file:
            file.write(new_env_content)

        print(f".env file updated to use the '{NEW_DB_NAME}' database.")
    else:
        print(f"Could not find the .env file at '{ENV_FILE_PATH}'.")

if __name__ == '__main__':
    update_database()

