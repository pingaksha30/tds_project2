#!/bin/bash

echo "Setting up LLM Quiz Project with AIPipe..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Chrome (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Installing Chrome and ChromeDriver..."
    sudo apt-get update
    sudo apt-get install -y chromium-browser chromium-chromedriver
fi

# Create .env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your credentials."
    echo ""
    echo "Get your AIPipe token from: https://aipipe.org/login"
fi

# Test the installation
echo "Testing installation..."
python -c "import fastapi, selenium, openai, pandas; print('All imports successful!')"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Get your AIPipe token from https://aipipe.org/login"
echo "2. Edit .env with your EMAIL, SECRET, and AIPIPE_TOKEN"
echo "3. Run: python main.py"
echo "4. Test with demo endpoint"
echo "5. Deploy to Render.com and submit to Google Form"
echo ""
echo "Your API will use GPT-5-nano through AIPipe proxy"
echo "Budget: 10 cents/week (free tier)"