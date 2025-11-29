import requests
import os
from dotenv import load_dotenv

load_dotenv()

SECRET = os.getenv("SECRET")
EMAIL = os.getenv("EMAIL")

# Test the demo-audio quiz
response = requests.post(
    "http://localhost:8000/quiz",
    json={
        "email": EMAIL,
        "secret": SECRET,
        "url": "https://tds-llm-analysis.s-anand.net/demo-audio"
    }
)

print("Response Status:", response.status_code)
print("Response:", response.json())
