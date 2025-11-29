#!/usr/bin/env python3
"""Test demo2 alphametic puzzle solving"""

import hashlib
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import os
import time

load_dotenv()
EMAIL = os.getenv("EMAIL")

def get_browser():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def fetch_quiz_page(url: str) -> str:
    driver = get_browser()
    try:
        # Add email parameter
        if '?' not in url:
            url = f"{url}?email={EMAIL}"
        
        driver.get(url)
        time.sleep(3)
        html_content = driver.page_source
        return html_content
    finally:
        driver.quit()

# Fetch the page
url = "https://tds-llm-analysis.s-anand.net/demo2"
print(f"Fetching: {url}")
html = fetch_quiz_page(url)

soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()

print(f"\n=== PAGE TEXT ===")
print(text[:500])

# Check for demo2 pattern
full_context_lower = html.lower()
if 'demo2' in text.lower() and 'emailnumber' in full_context_lower:
    print(f"\n🔑 Demo2 canvas puzzle detected - extracting key from email")
    
    # Calculate emailNumber: first 4 hex of SHA1(email) as integer
    sha1_hash = hashlib.sha1(EMAIL.encode()).hexdigest()
    email_number = int(sha1_hash[:4], 16)
    print(f"📧 Email: {EMAIL}")
    print(f"🔢 EmailNumber (first 4 hex of SHA1): {email_number}")
    
    # Calculate key: (emailNumber * 7919 + 12345) mod 1e8
    key = (email_number * 7919 + 12345) % int(1e8)
    key_str = str(key).zfill(8)
    print(f"🔑 Calculated key: {key_str}")
    
    # Verify
    letters = ["F", "O", "R", "K", "L", "I", "M", "E"]
    mapping = dict(zip(letters, key_str))
    fork = int(mapping['F'] + mapping['O'] + mapping['R'] + mapping['K'])
    lime = int(mapping['L'] + mapping['I'] + mapping['M'] + mapping['E'])
    total = fork + lime
    print(f"✅ Verification: FORK({fork}) + LIME({lime}) = {total}")
    print(f"\n📤 Answer: {key_str}")
else:
    print("\n❌ Demo2 pattern not detected")
    print(f"Has 'demo2': {'demo2' in text.lower()}")
    print(f"Has 'emailnumber': {'emailnumber' in full_context_lower()}")
