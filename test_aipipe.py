#!/usr/bin/env python3
"""Test script to verify AIPipe connection"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

if not AIPIPE_TOKEN:
    print("❌ AIPIPE_TOKEN not found in .env file")
    print("Get your token from: https://aipipe.org/login")
    exit(1)

print("Testing AIPipe connection with GPT-5-nano...")

try:
    client = OpenAI(
        api_key=AIPIPE_TOKEN,
        base_url="https://aipipe.org/openrouter/v1"
    )
    
    response = client.chat.completions.create(
        model="openai/gpt-5-nano",
        messages=[
            {"role": "user", "content": "Say 'Hello from GPT-5-nano!' if you can read this."}
        ],
        max_tokens=50
    )
    
    result = response.choices[0].message.content
    print(f"✅ Success! Response: {result}")
    print("\nAIPipe is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure:")
    print("1. You have a valid AIPIPE_TOKEN in .env")
    print("2. You're connected to the internet")
    print("3. Your token hasn't expired")