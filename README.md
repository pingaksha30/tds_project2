# tds_project2
Setup Instructions
1. Get Your AIPipe Token
Visit https://aipipe.org/login
Sign in with your Google account
Copy your AIPipe token from the dashboard
Paste it in your .env file
2. Install Dependencies
bash pip install -r requirements.txt

3. Install Chrome and ChromeDriver
Ubuntu or Debian: bash sudo apt-get update sudo apt-get install -y chromium-browser chromium-chromedriver

macOS: bash brew install --cask google-chrome brew install chromedriver

4. Configure Environment Variables
bash cp .env.example .env

Edit .env with your actual values
5. Run Locally
bash python main.py

The API will be available at http://localhost:8000

6. Test Your Endpoint
bash curl -X POST http://localhost:8000/quiz
-H "Content-Type: application/json"
-d '{ "email": "your-email@example.com", "secret": "your_secret", "url": "https://tds-llm-analysis.s-anand.net/demo" }'

7. Deploy
Push code to GitHub with MIT LICENSE
Deploy on Render.com or similar platform
Add environment variables in deployment settings
Use your deployed URL + /quiz for the Google Form
Key Features
Uses AIPipe proxy for GPT-5-nano access
Headless browser for JavaScript rendering
File download and processing (PDF, CSV, JSON)
Data analysis with pandas
Visualization with matplotlib/seaborn
AI-powered quiz solving
Automatic answer submission
Multi-stage quiz handling
Time-aware processing (under 3 minutes)
AIPipe Usage
This project uses AIPipe as a proxy to access GPT-5-nano:

Base URL: https://aipipe.org/openrouter/v1
Model: openai/gpt-5-nano
Budget: 10 cents per week (free tier)
Troubleshooting
ChromeDriver issues: bash pip install webdriver-manager

Timeout errors:

Check internet connection
Increase timeout values in httpx calls
Memory issues:

Monitor file sizes (max 1MB for submissions)
Use data streaming for large files
