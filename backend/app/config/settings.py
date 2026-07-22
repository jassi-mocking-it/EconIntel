from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Read the FRED API key
FRED_API_KEY = os.getenv("FRED_API_KEY")