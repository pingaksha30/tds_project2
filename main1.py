from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
import httpx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import base64
from typing import Optional, Any
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import PyPDF2
import re

load_dotenv()

app = FastAPI()

# Debug info captured from Python analysis (populated by analyze_question_and_data)
ANALYSIS_DEBUG = {}

# Environment variables
SECRET = os.getenv("SECRET")
EMAIL = os.getenv("EMAIL")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

# Initialize OpenAI client with aipipe base URL
client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)

# Load Whisper model ONCE at startup (not per-request) for speed
WHISPER_MODEL = None
def get_whisper_model():
    """Load Whisper model once and cache it"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            import whisper
            print("🔧 Loading Whisper 'tiny' model at startup...")
            WHISPER_MODEL = whisper.load_model("tiny")
            print("✅ Whisper model loaded and cached")
        except ImportError:
            print("⚠️  whisper package not installed")
            return None
    return WHISPER_MODEL

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

class AnswerSubmission(BaseModel):
    email: str
    secret: str
    url: str
    answer: Any

@app.on_event("startup")
async def startup_event():
    """Load Whisper model at startup for faster transcription"""
    print("\n🚀 Starting quiz solver...")
    get_whisper_model()  # Pre-load Whisper model
    print("✅ Ready to solve quizzes!\n")

def get_browser():
    """Initialize headless Chrome browser"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def fetch_quiz_page(url: str, add_email: bool = False) -> str:
    """Fetch and render JavaScript-enabled quiz page"""
    driver = get_browser()
    try:
        # Add email parameter if required
        if add_email and '?' not in url:
            url = f"{url}?email={EMAIL}"
        elif add_email and '?' in url:
            url = f"{url}&email={EMAIL}"
        
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        # Wait longer for JavaScript to render puzzle
        time.sleep(3)
        html_content = driver.page_source
        return html_content
    finally:
        driver.quit()

def download_file(url: str) -> bytes:
    """Download file from URL"""
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content

def extract_pdf_text(pdf_content: bytes, page_num: Optional[int] = None) -> str:
    """Extract text from PDF"""
    pdf_file = BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    if page_num is not None:
        return pdf_reader.pages[page_num - 1].extract_text()
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def transcribe_audio(audio_content: bytes, filename: str) -> str:
    """Transcribe audio file using local Whisper model"""
    import tempfile
    import subprocess
    
    # Get cached Whisper model
    model = get_whisper_model()
    if model is None:
        print(f"  ⚠️  whisper package not installed. Install with: pip install openai-whisper")
        return ""
    
    # Save to temp file with original extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(audio_content)
        tmp_path = tmp.name
    
    try:
        # Convert opus/ogg/weba to mp3 for better compatibility (FAST conversion)
        if filename.endswith(('.opus', '.ogg', '.weba', '.aac')):
            print(f"  Converting {filename} to MP3 (fast mode)...")
            mp3_path = tmp_path.replace(os.path.splitext(tmp_path)[1], '.mp3')
            
            try:
                # Use faster ffmpeg settings: lower quality but MUCH faster
                subprocess.run(
                    ['ffmpeg', '-i', tmp_path, '-acodec', 'libmp3lame', '-ab', '64k', '-ar', '16000', mp3_path, '-y'],
                    check=True, 
                    capture_output=True,
                    text=True
                )
                os.unlink(tmp_path)
                tmp_path = mp3_path
                print(f"  ✓ Converted to MP3")
            except FileNotFoundError:
                print("  ⚠️  ffmpeg not found, trying direct transcription...")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  Conversion failed, trying direct transcription...")
        
        # Use the CACHED model (already loaded at startup)
        print(f"  Transcribing {os.path.basename(tmp_path)} with cached model...")
        result = model.transcribe(tmp_path)
        transcription_text = result["text"].strip()
        
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        print(f"  ✓ Transcription successful: '{transcription_text[:100]}'")
        return transcription_text
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        print(f"  Audio transcription error: {e}")
        print(f"  Error type: {type(e).__name__}")
        return ""

def analyze_image(image_content: bytes, question: str) -> str:
    """Analyze image using GPT-4 Vision"""
    try:
        # Encode image to base64
        img_base64 = base64.b64encode(image_content).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image and answer: {question}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Image analysis error: {e}")
        return f"Image analysis failed: {e}"

def create_visualization(data: pd.DataFrame, viz_type: str) -> str:
    """Create visualization and return as base64 (matplotlib only, no seaborn)"""
    plt.figure(figsize=(10, 6))
    
    if viz_type == "bar":
        data.plot(kind='bar')
    elif viz_type == "line":
        data.plot(kind='line')
    elif viz_type == "scatter":
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    elif viz_type == "heatmap":
        # Simple heatmap using matplotlib (no seaborn)
        plt.imshow(data.values, aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(data.columns)), data.columns, rotation=45)
        plt.yticks(range(len(data.index)), data.index)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def solve_quiz_with_gpt(quiz_html: str, downloaded_files: dict = None) -> Any:
    """Use GPT-5-nano to analyze quiz and generate answer"""
    
    soup = BeautifulSoup(quiz_html, 'html.parser')
    question_text = soup.get_text(separator="\n", strip=True)
    
    print(f"\n--- QUESTION TEXT ---\n{question_text[:500]}...\n")
    
    # Check if audio transcriptions exist - needed for data provision logic
    has_audio = any(
        any(ext in filename for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba'])
        for filename in downloaded_files.keys()
        if isinstance(downloaded_files.get(filename), str) and downloaded_files.get(filename)
    ) if downloaded_files else False
    
    context = f"""You are solving a data analysis quiz. Here is the complete question page:

{question_text}

"""
    
    if downloaded_files:
        # First, check if there are any audio transcriptions and prioritize them
        audio_transcriptions = []
        for filename, content in downloaded_files.items():
            if not filename.endswith('_dataframe') and not filename.endswith('_image') and isinstance(content, str):
                is_audio = any(ext in filename for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba'])
                if is_audio and content:
                    audio_transcriptions.append((filename, content))
        
        # If audio exists, emphasize it at the top
        if audio_transcriptions:
            context += "\n\n⚠️ CRITICAL AUDIO INSTRUCTIONS - FOLLOW THESE EXACTLY:\n"
            for filename, content in audio_transcriptions:
                context += f"\n--- AUDIO TRANSCRIPTION from {filename} ---\n{content}\n"
                context += "\n>>> The audio instructions above are MANDATORY. Follow them precisely when analyzing the data below. <<<\n"
                print(f"  ⚠️  AUDIO INSTRUCTIONS: {content[:150]}...")
        
        context += "\n\nI have downloaded and processed these files for you:\n"
        for filename, content in downloaded_files.items():
            # Skip dataframe and image objects in the prompt
            if filename.endswith('_dataframe') or filename.endswith('_image'):
                continue
            if isinstance(content, str):
                # Check if this is an audio transcription
                is_audio = any(ext in filename for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba'])
                
                if is_audio:
                    # Already added at top, skip here
                    print(f"  - {filename} (transcription: {len(content)} chars)")
                elif len(content) > 5000:
                    context += f"\n--- {filename} (first 5000 chars) ---\n{content[:5000]}...\n"
                    print(f"  - {filename} (truncated)")
                else:
                    context += f"\n--- {filename} ---\n{content}\n"
                    print(f"  - {filename}")
            else:
                context += f"\n--- {filename} ---\n{str(content)}\n"
                print(f"  - {filename} (converted to string)")
        
        # Add DataFrame data - for audio quizzes, provide ALL values so LLM can apply cutoffs
        for filename, content in downloaded_files.items():
            if filename.endswith('_dataframe'):
                df = content
                # Work with positional first column to avoid header issues
                col_series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                col_label = df.columns[0]
                
                # Check if this is an audio quiz - if so, provide complete data
                if has_audio:
                    context += f"\n--- {filename.replace('_dataframe', '')} (CSV Complete Data) ---\n"
                    context += f"Column: {col_label}\n"
                    context += f"Total rows: {len(df)}\n"
                    context += f"ALL VALUES (so you can apply cutoff operations):\n"
                    context += f"{col_series.dropna().tolist()}\n"
                    context += f"\nStatistics for reference:\n"
                    context += f"Sum of ALL values: {int(col_series.sum())}\n"
                    context += f"Mean: {col_series.mean():.2f}, Min: {col_series.min()}, Max: {col_series.max()}\n"
                else:
                    # No audio - just provide summary (Python preprocessing will handle calculations)
                    context += f"\n--- {filename.replace('_dataframe', '')} (CSV Summary) ---\n"
                    context += f"Total rows: {len(df)}, Column: {col_label}\n"
                    context += f"Sum: {int(col_series.sum())}, Mean: {col_series.mean():.2f}\n"
                    context += f"Min: {col_series.min()}, Max: {col_series.max()}\n"
                    context += f"First 5 values: {col_series.head().tolist()}\n"
                    context += f"Last 5 values: {col_series.tail().tolist()}\n"
        
        print(f"\n--- FILES PROVIDED TO GPT ---")
        for filename, content in downloaded_files.items():
            if filename.endswith('_dataframe'):
                print(f"  - {filename.replace('_dataframe', '')} (CSV - {len(content)} rows)")
            elif filename.endswith('_image'):
                print(f"  - {filename.replace('_image', '')} (Image - stored for vision)")
            elif isinstance(content, str):
                is_audio = any(ext in filename for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba'])
                if is_audio and content:
                    print(f"  - {filename} (TRANSCRIPTION: '{content[:100]}...')")
                else:
                    print(f"  - {filename} (text, {len(content)} chars)")
    
    if has_audio:
        context += """\n\n⚠️ CRITICAL: Audio transcription contains specific instructions. You MUST follow them exactly.

You are an expert data analyst. The audio transcription above contains the PRIMARY INSTRUCTIONS for this task.

IMPORTANT RULES:
1. Read the audio transcription carefully - it tells you EXACTLY what to calculate
2. If audio mentions "greater than or equal" or "at least", use >= operator
3. If audio mentions "greater than" (without "equal"), use > operator  
4. If audio mentions "less than or equal" or "at most", use <= operator
5. If audio mentions "less than" (without "equal"), use < operator
6. The audio instructions override any ambiguity in the question text

For CSV analysis WITH CUTOFF:
- I have provided you with ALL the values from the CSV file above
- You can see the complete list of numbers
- Apply the cutoff filter yourself based on the audio instructions
- Example: If audio says "sum all values greater than or equal to 25514"
  1. Look at the list of ALL VALUES provided
  2. Filter: keep only values >= 25514
  3. Sum those filtered values
  4. Return the sum

For text/code extraction:
- Extract secret codes, passwords, or specific values mentioned
- Look for patterns in the data

Provide ONLY the final answer:
- Numbers: return just the number (e.g., 498500)
- Text: return just the text (e.g., "secret_xyz")
- Base64 images: return the full data URI
- JSON: return the JSON object
- Boolean: return true or false

DO NOT include explanations. Your answer:"""
    else:
        context += """\n\nYou are an expert data analyst. Analyze the question and data carefully.

For alphametic/cryptarithmetic puzzles:
- Solve puzzles like "SEND + MORE = MONEY"
- Return a JSON mapping of letters to digits (e.g., {"S": 9, "E": 5, ...})
- Leading letters cannot be 0

For CSV analysis:
- If there's a cutoff mentioned, calculate sum/count of values ABOVE that cutoff
- For calculations, use the statistics I've provided above
- Look for keywords: sum, count, average, mean, median, min, max, filter, aggregate

For text/code extraction:
- Extract secret codes, passwords, or specific values mentioned
- Look for patterns in the data

For visualizations:
- If asked to create a chart/graph, return a base64 data URI
- Format: data:image/png;base64,[base64_string]

Provide ONLY the final answer:
- Numbers: return just the number (e.g., 498500)
- Text: return just the text (e.g., "secret_xyz")
- Base64 images: return the full data URI
- JSON: return the JSON object (for alphametic solutions)
- Boolean: return true or false

NEVER return error objects or status messages. If you cannot solve it, return null.

DO NOT include explanations. Your answer:"""
    
    response = client.chat.completions.create(
        model="openai/gpt-5-nano",
        messages=[
            {"role": "user", "content": context}
        ],
        max_tokens=4000
    )
    
    response_text = response.choices[0].message.content.strip()
    print(f"GPT Response: {response_text}")
    
    # Handle empty or null response
    if not response_text or response_text.lower() == 'null':
        print("⚠️  GPT could not determine answer")
        return None
    
    # Try to parse as JSON first
    try:
        parsed = json.loads(response_text)
        # Reject error objects
        if isinstance(parsed, dict) and 'error' in parsed:
            print("⚠️  GPT returned error object, rejecting")
            return None
        if isinstance(parsed, dict) and 'status' in parsed and parsed.get('status') == 'error':
            print("⚠️  GPT returned error status, rejecting")
            return None
        return parsed
    except:
        pass
    
    # Try to parse as number
    try:
        # Remove any surrounding quotes
        clean_text = response_text.strip('"\'')
        if '.' in clean_text:
            return float(clean_text)
        return int(clean_text)
    except:
        pass
    
    # Return as string if nothing else works
    return response_text

def process_quiz(url: str) -> tuple[str, Any]:
    """Main quiz processing logic"""
    quiz_html = fetch_quiz_page(url)
    soup = BeautifulSoup(quiz_html, 'html.parser')
    
    # Check if page requires email parameter
    page_text = soup.get_text().lower()
    if 'add ?email=' in page_text or 'enable javascript' in page_text:
        print(f"  📧 Page requires email parameter, retrying...")
        quiz_html = fetch_quiz_page(url, add_email=True)
        soup = BeautifulSoup(quiz_html, 'html.parser')
    
    # Extract submit URL - look for it in the page text
    submit_url = None
    text_content = soup.get_text()
    
    # Look for explicit submit URL mentions in text
    submit_pattern = r'(https?://[^\s<>"{}|\\^\[\]`]*submit[^\s<>"{}|\\^\[\]`]*)'
    submit_matches = re.findall(submit_pattern, text_content, re.IGNORECASE)
    
    if submit_matches:
        # Take the first submit URL that doesn't have query parameters (it's the base endpoint)
        for match in submit_matches:
            if '?' not in match:
                submit_url = match
                break
        # If all have query params, take the first one
        if not submit_url:
            submit_url = submit_matches[0]
    
    # Fallback: find all URLs and look for submit
    if not submit_url:
        url_pattern = r'https?://[^\s<>"{}|\\^\[\]`]+'
        urls = re.findall(url_pattern, text_content)
        for found_url in urls:
            if 'submit' in found_url.lower() and '?' not in found_url:
                submit_url = found_url
                break
    
    # Last resort: derive from quiz URL
    if not submit_url:
        base_domain = '/'.join(url.split('/')[:3])
        submit_url = f"{base_domain}/submit"
    
    print(f"Extracted submit URL: {submit_url}")
    
    downloaded_files = {}
    file_links = soup.find_all('a', href=True)
    print(f"Found {len(file_links)} links on page")
    
    # Also search for audio/video tags in the HTML
    audio_tags = soup.find_all(['audio', 'video', 'source'])
    print(f"Found {len(audio_tags)} audio/video elements")
    
    for tag in audio_tags:
        src = tag.get('src')
        if src:
            print(f"Found media source: {src}")
            if not src.startswith('http'):
                base_url = '/'.join(url.split('/')[:3])
                if not src.startswith('/'):
                    src = '/' + src
                src = base_url + src
            
            # Create a fake link object to process it
            class FakeLink:
                def __init__(self, href):
                    self.attrs = {'href': href}
                def __getitem__(self, key):
                    return self.attrs.get(key)
            
            file_links.append(FakeLink(src))
            print(f"Added audio/video to download list: {src}")
    
    for link in file_links:
        file_url = link['href']
        print(f"Processing link: {file_url}")
        
        if not file_url.startswith('http'):
            base_url = '/'.join(url.split('/')[:3])
            # Ensure proper slash between domain and path
            if not file_url.startswith('/'):
                file_url = '/' + file_url
            file_url = base_url + file_url
        
        # Skip submit URLs - they're endpoints, not downloadable files
        if 'submit' in file_url.lower():
            print(f"Skipping submit URL: {file_url}")
            continue
        
        try:
            print(f"Downloading: {file_url}")
            file_content = download_file(file_url)
            filename = file_url.split('/')[-1].split('?')[0]  # Remove query params from filename
            
            if filename.endswith('.pdf'):
                downloaded_files[filename] = extract_pdf_text(file_content)
                print(f"✓ Extracted PDF: {filename} ({len(downloaded_files[filename])} chars)")
            elif filename.endswith('.csv'):
                # Read CSV and ensure ALL rows are loaded (no limits)
                # Initial read
                df = pd.read_csv(BytesIO(file_content))
                # If the CSV has a single column and the column name looks numeric,
                # it's likely pandas used the first data row as header. Re-read
                # with header=None to preserve all rows.
                forced_header = False
                if len(df.columns) == 1:
                    col0 = str(df.columns[0]).strip()
                    if col0.isdigit():
                        try:
                            df = pd.read_csv(BytesIO(file_content), header=None)
                            forced_header = True
                        except Exception:
                            pass

                print(f"✓ Loaded CSV: {filename} ({len(df)} rows, {len(df.columns)} cols)")
                print(f"  Columns: {list(df.columns)}")
                # Use iloc to access first column safely
                try:
                    col_series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                    print(f"  Total sum of column: {int(col_series.sum())}")
                    print(f"  First 3 values: {col_series.head(3).tolist()}")
                    print(f"  Last 3 values: {col_series.tail(3).tolist()}")
                except Exception:
                    print(f"  Could not summarize CSV column")

                # Store actual dataframe and header-forced flag
                downloaded_files[f"{filename}_dataframe"] = df
                if forced_header:
                    downloaded_files[f"{filename}_dataframe_header_forced"] = True
            elif filename.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba')):
                print(f"\n⚠️  AUDIO FILE DETECTED - Transcribing before analysis...")
                transcription = transcribe_audio(file_content, filename)
                if transcription:
                    downloaded_files[filename] = transcription
                    print(f"✓ Transcribed audio: {filename}")
                    print(f"  📝 TRANSCRIPTION: {transcription}")
                    print(f"  ⚠️  This transcription will be prioritized in LLM prompt\n")
                else:
                    print(f"⚠️  Audio transcription returned empty")
                    downloaded_files[filename] = ""
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                print(f"  Image file detected, storing for analysis...")
                downloaded_files[f"{filename}_image"] = file_content
                print(f"✓ Loaded image: {filename} ({len(file_content)} bytes)")
            elif filename.endswith(('.txt', '.json')):
                downloaded_files[filename] = file_content.decode('utf-8')
                print(f"✓ Loaded text file: {filename} ({len(downloaded_files[filename])} chars)")
            else:
                # For files without extensions or HTML, use Selenium to render JavaScript
                try:
                    text_content = file_content.decode('utf-8')
                    # Check if it contains JavaScript that needs rendering
                    if '<script' in text_content.lower():
                        print(f"  Detected JavaScript, rendering with Selenium...")
                        rendered_html = fetch_quiz_page(file_url)
                        rendered_soup = BeautifulSoup(rendered_html, 'html.parser')
                        rendered_text = rendered_soup.get_text(separator="\n", strip=True)
                        downloaded_files[filename] = rendered_text
                        print(f"✓ Rendered with JS: {filename} ({len(rendered_text)} chars)")
                        print(f"  Content preview: {rendered_text[:200]}")
                    else:
                        downloaded_files[filename] = text_content
                        print(f"✓ Decoded as text: {filename} ({len(text_content)} chars)")
                        print(f"  Content: {text_content[:200]}")
                except:
                    downloaded_files[filename] = f"Binary file: {len(file_content)} bytes"
                    print(f"✓ Downloaded binary: {filename} ({len(file_content)} bytes)")
        except Exception as e:
            print(f"✗ Error downloading {file_url}: {e}")
    
    # Check if audio files are present
    has_audio = any(
        any(ext in filename for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba'])
        for filename in downloaded_files.keys()
        if isinstance(downloaded_files.get(filename), str) and downloaded_files.get(filename)
    )
    
    print(f"\n--- ANALYZING QUESTION ---")
    
    # ALWAYS try Python preprocessing first (it's more reliable for calculations)
    print(f"🔧 Using Python preprocessing tools for data analysis...")
    
    # Pass the full HTML (not just text) so JavaScript patterns can be detected
    full_html = quiz_html
    answer = analyze_question_and_data(full_html, downloaded_files, current_url=url)
    
    # Only fall back to GPT if Python analysis completely failed
    if answer is None:
        print("⚠️  Python analysis could not determine answer")
        if has_audio:
            print("📤 Falling back to LLM for audio interpretation...")
        else:
            print("📤 Falling back to LLM (last resort)...")
        answer = solve_quiz_with_gpt(quiz_html, downloaded_files)
    else:
        print(f"✅ Python preprocessing successfully computed answer")
    
    print(f"Final Answer: {answer} (type: {type(answer).__name__})")
    
    # Ensure answer is a simple type (not dict/list unless it's the actual answer)
    if isinstance(answer, dict) and 'status' in answer and answer.get('status') == 'error':
        print(f"  ⚠️  Answer is an error object, this will fail submission")
        print(f"  🔄 Returning None to indicate failure")
        answer = None
    
    # Print debug details captured during analysis
    if ANALYSIS_DEBUG:
        try:
            print(f"ANALYSIS_DEBUG: {json.dumps(ANALYSIS_DEBUG, indent=2)}")
        except Exception:
            print(f"ANALYSIS_DEBUG: {ANALYSIS_DEBUG}")

    return submit_url, answer

def solve_alphametic(word1: str, word2: str, result_word: str) -> Optional[dict]:
    """Solve alphametic/cryptarithmetic puzzles like SEND + MORE = MONEY"""
    from itertools import permutations
    
    # Get unique letters
    letters = set(word1 + word2 + result_word)
    if len(letters) > 10:
        return None  # Can't map to digits 0-9
    
    # Leading letters cannot be 0
    leading_letters = {word1[0], word2[0], result_word[0]}
    
    # Try all permutations of digits
    for perm in permutations(range(10), len(letters)):
        mapping = dict(zip(letters, perm))
        
        # Check if leading letters are not 0
        if any(mapping[letter] == 0 for letter in leading_letters):
            continue
        
        # Convert words to numbers
        num1 = int(''.join(str(mapping[c]) for c in word1))
        num2 = int(''.join(str(mapping[c]) for c in word2))
        result_num = int(''.join(str(mapping[c]) for c in result_word))
        
        # Check if equation holds
        if num1 + num2 == result_num:
            return mapping
    
    return None


def extract_cutoff_operator_and_value(text: str) -> tuple[Optional[str], Optional[int]]:
    """Extract a cutoff operator and integer value from natural language text.

    Returns a tuple (operator, value) where operator is one of '>=', '>', '<=', '<',
    or None if not found. Value is an int or None.
    """
    patterns = [
        (r'greater than or equal to\s+([0-9,]+)', '>='),
        (r'greater than or equals\s+([0-9,]+)', '>='),
        (r'>=\s*([0-9,]+)', '>='),
        (r'at least\s+([0-9,]+)', '>='),
        (r'greater than\s+([0-9,]+)', '>'),
        (r'>\s*([0-9,]+)', '>'),
        (r'less than or equal to\s+([0-9,]+)', '<='),
        (r'less than or equals\s+([0-9,]+)', '<='),
        (r'<=\s*([0-9,]+)', '<='),
        (r'at most\s+([0-9,]+)', '<='),
        (r'less than\s+([0-9,]+)', '<'),
        (r'<\s*([0-9,]+)', '<'),
        (r'([0-9,]{2,})\s*(?:or more|and above|and over)', '>='),
        (r'([0-9,]{2,})\s*(?:or less|and below|and under)', '<='),
    ]

    for pat, op in patterns:
        m = re.search(pat, text)
        if m:
            raw = m.group(1).replace(',', '')
            try:
                return op, int(raw)
            except Exception:
                continue

    return None, None

def analyze_question_and_data(question_text: str, downloaded_files: dict, current_url: Optional[str] = None) -> Any:
    """Analyze question and perform calculations in Python - PRIMARY analysis method"""
    print(f"🔍 Python Analysis Started")
    question_lower = question_text.lower()
    
    # Combine question text with any audio transcriptions FIRST
    full_context = question_text
    audio_transcription_available = False
    
    for filename, content in downloaded_files.items():
        if not filename.endswith('_dataframe') and not filename.endswith('_image') and isinstance(content, str):
            if any(ext in filename for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.aac', '.weba']):
                # Only add if transcription was successful (not empty and not error message)
                if content and not content.startswith('[Audio transcription failed'):
                    full_context += "\n" + content
                    audio_transcription_available = True
                    print(f"  ✓ Added audio transcription to context: {content[:100]}")
                elif not content:
                    print(f"  ⚠️  Audio file {filename} detected but transcription unavailable")
    
    full_context_lower = full_context.lower()
    
    # PRIORITY 0: Extract JavaScript source code patterns (for canvas-based puzzles)
    # Check if question contains JavaScript with key logic patterns
    if '<script' in question_text and 'emailnumber' in full_context_lower:
        print(f"  🔎 Stage 0: JavaScript puzzle detected, extracting logic...")
        # This will be caught in Stage 1.5 alphametic detection
    
    # PRIORITY 1: Check for explicit answer in question (like "answer": "anything you want")
    # But prioritize scraped data over question placeholders
    print(f"  🔎 Stage 1: Checking for explicit answers in question...")
    answer_pattern = re.search(r'["\']answer["\']\s*:\s*["\']([^"\'])+["\']', question_text, re.IGNORECASE)
    potential_answer_from_question = None
    if answer_pattern:
        result = answer_pattern.group(1)
        # Check if it's NOT a placeholder
        placeholder_phrases = ['...', '…', 'your answer', 'your_answer', 'you scraped', 'you extracted', 'you calculated', 'you found']
        is_placeholder = any(phrase in result.lower() for phrase in placeholder_phrases)
        
        if result and not is_placeholder:
            print(f"    ✓ Found potential answer in question: {result}")
            potential_answer_from_question = result
        elif is_placeholder:
            print(f"    ⊗ Skipping placeholder in question: {result}")

    # Stage 1.2: Quick API data handler - detect instructions to call an API and sum 'values'
    print(f"  🔎 Stage 1.2: Checking for API data puzzles...")
    try:
        combined_text = question_text + '\n' + '\n'.join(k for k in downloaded_files.keys()) if downloaded_files else question_text
    except Exception:
        combined_text = question_text
    combined_lower = combined_text.lower()
    if ("api" in combined_lower and ("values" in combined_lower or '"values"' in combined_text) and "sum" in combined_lower):
        print("    🔎 Looks like an API data puzzle (values array + sum). Attempting to find API URL...")
        api_url = None
        # prefer absolute URL in question_text/html
        m = re.search(r'https?://[^\s\"\']+/api[^\s\"\']*', question_text)
        if m:
            api_url = m.group(0)
        else:
            # try to find a path like /api or /api/data and resolve relative to current_url
            m2 = re.search(r'(/api[^\s\"\']*)', question_text)
            if m2 and current_url:
                base = '/'.join(current_url.split('/')[:3])
                path = m2.group(1)
                api_url = base + path if path.startswith('/') else base + '/' + path

        if api_url:
            print(f"    🔗 Found API URL candidate: {api_url}")
            params = {"email": EMAIL, "secret": SECRET}
            try:
                with httpx.Client(timeout=15.0) as client:
                    r = client.get(api_url, params=params)
                    r.raise_for_status()
                    json_data = r.json()
                if isinstance(json_data, dict) and 'values' in json_data and isinstance(json_data['values'], list):
                    # convert to numbers where possible
                    nums = []
                    for v in json_data['values']:
                        try:
                            nums.append(float(v))
                        except Exception:
                            try:
                                nums.append(int(v))
                            except Exception:
                                pass
                    if nums:
                        s = int(sum(nums))
                        print(f"    ✅ Fetched API data; computed sum(values) = {s}")
                        return s
                    else:
                        print("    ⚠️ API 'values' array exists but contains no numeric entries")
                else:
                    print("    ⚠️ API response did not contain a 'values' list or unexpected format")
            except Exception as e:
                print(f"    ⚠️ API fetch failed: {type(e).__name__}: {e}")
        else:
            print("    ⚠️ No API URL found in question text; skipping API attempt")
    
    # PRIORITY 1.5: Check for alphametic/cryptarithmetic puzzles
    print(f"  🔎 Stage 1.5: Checking for alphametic puzzles...")
    if 'alphametic' in full_context_lower or 'cryptarithmetic' in full_context_lower:
        print(f"    🧩 Alphametic puzzle detected")
        
        # Check if this is a canvas-based puzzle with emailNumber logic (like demo2)
        # Pattern: JavaScript mentions "emailNumber", "SHA1", and key calculation
        if 'emailnumber' in full_context_lower and ('sha1' in full_context_lower or 'sha-1' in full_context_lower):
            print(f"    🔑 Email-based key puzzle detected")
            import hashlib
            
            # Try to extract the formula from JavaScript if present
            # Look for patterns like: (emailNumber * XXXX + YYYY) mod ZZZZ
            multiplier = 7919  # default
            offset = 12345     # default
            modulo = int(1e8)  # default
            
            # Try to extract from source
            mult_pattern = re.search(r'emailnumber\s*\*\s*(\d+)', full_context_lower)
            if mult_pattern:
                multiplier = int(mult_pattern.group(1))
                print(f"    📐 Extracted multiplier: {multiplier}")
            
            offset_pattern = re.search(r'\+\s*(\d+)\s*\)', full_context_lower)
            if offset_pattern:
                offset = int(offset_pattern.group(1))
                print(f"    📐 Extracted offset: {offset}")
            
            # Calculate emailNumber: first 4 hex of SHA1(email) as integer
            sha1_hash = hashlib.sha1(EMAIL.encode()).hexdigest()
            email_number = int(sha1_hash[:4], 16)
            print(f"    📧 Email: {EMAIL}")
            print(f"    🔢 EmailNumber (first 4 hex of SHA1): {email_number}")
            
            # Calculate key: (emailNumber * multiplier + offset) mod modulo
            key = (email_number * multiplier + offset) % modulo
            key_str = str(key).zfill(8)  # Ensure 8 digits with leading zeros
            print(f"    🔑 Calculated key: {key_str}")
            
            # Try to verify if there's an equation in the puzzle
            letters_pattern = re.search(r'letters\s*=\s*\[([^\]]+)\]', full_context, re.IGNORECASE)
            if letters_pattern:
                letters_str = letters_pattern.group(1).replace('"', '').replace("'", '').replace(' ', '')
                letters = letters_str.split(',')
                print(f"    📝 Found letters: {letters}")
                
                mapping = dict(zip(letters, key_str))
                # Try to find the equation (e.g., FORK + LIME)
                equation_words = re.findall(r'\b[A-Z]{4,}\b', full_context.upper())
                if len(equation_words) >= 2:
                    word1, word2 = equation_words[0], equation_words[1]
                    num1 = int(''.join(mapping.get(c, '0') for c in word1))
                    num2 = int(''.join(mapping.get(c, '0') for c in word2))
                    print(f"    ✅ Verification: {word1}({num1}) + {word2}({num2}) = {num1 + num2}")
            
            print(f"    📤 Returning key: {key_str}")
            return key_str
        
        # Standard alphametic puzzle (SEND + MORE = MONEY)
        equation_pattern = re.search(r'([A-Z]+)\s*\+\s*([A-Z]+)\s*=\s*([A-Z]+)', full_context, re.IGNORECASE)
        if equation_pattern:
            word1, word2, result_word = equation_pattern.groups()
            word1, word2, result_word = word1.upper(), word2.upper(), result_word.upper()
            print(f"    📝 Found equation: {word1} + {word2} = {result_word}")
            
            # Solve alphametic
            solution = solve_alphametic(word1, word2, result_word)
            if solution:
                print(f"    ✅ Solved alphametic: {solution}")
                return solution
            else:
                print(f"    ⚠️  Could not solve alphametic")
    
    # PRIORITY 1.6: Check for checksum/hash puzzles
    print(f"  🔎 Stage 1.6: Checking for checksum puzzles...")
    if ('checksum' in full_context_lower or 'hash' in full_context_lower) and ('sha256' in full_context_lower or 'sha-256' in full_context_lower):
        print(f"    🔐 Checksum puzzle detected")
        
        # Look for patterns that indicate we need to:
        # 1. Use a previous key/answer
        # 2. Append/combine with a blob/salt
        # 3. Compute SHA256
        # 4. Return first N hex characters
        
        # Extract blob/salt pattern (hex string) - try multiple patterns
        blob = None
        
        # Pattern 0: Search in raw question_text (HTML) first for <code> tags containing hex
        if not blob:
            code_match = re.search(r'<code>\s*([a-fA-F0-9]{6,})\s*</code>', question_text, re.IGNORECASE)
            if code_match:
                blob = code_match.group(1).strip()
                print(f"    ✓ Found blob in <code> tag: {blob}")
        
        # Pattern 1: "Blob:" followed by hex on next line or same line
        if not blob:
            blob_pattern = re.search(r'blob\s*:\s*\n?\s*([a-fA-F0-9]{6,})', full_context, re.IGNORECASE | re.MULTILINE)
            if blob_pattern:
                blob = blob_pattern.group(1).strip()
        
        # Pattern 2: "append" followed by hex
        if not blob:
            blob_pattern = re.search(r'append\s+(?:the\s+)?(?:blob\s+)?(?:below\s+)?(?:exactly\s*[,:])?\s*([a-fA-F0-9]{6,})', full_context, re.IGNORECASE | re.MULTILINE)
            if blob_pattern:
                blob = blob_pattern.group(1).strip()
        
        # Pattern 3: "salt:" followed by hex
        if not blob:
            blob_pattern = re.search(r'salt\s*:\s*\n?\s*([a-fA-F0-9]{6,})', full_context, re.IGNORECASE | re.MULTILINE)
            if blob_pattern:
                blob = blob_pattern.group(1).strip()
        
        # Pattern 4: Look for standalone hex string (6+ chars) after keywords in cleaned text
        if not blob:
            # Find text after "blob" keyword and look for hex in next 200 chars
            blob_section = re.search(r'blob[:\s]+(.{1,200})', full_context, re.IGNORECASE | re.DOTALL)
            if blob_section:
                # Extract hex string from that section (remove HTML tags first)
                section_text = blob_section.group(1)
                section_clean = re.sub(r'<[^>]*>', '', section_text)
                hex_match = re.search(r'\b([a-fA-F0-9]{6,})\b', section_clean)
                if hex_match:
                    blob = hex_match.group(1).strip()
        
        # Pattern 5: Last resort - search for any hex string (6+ chars) in raw question_text after "blob"
        if not blob:
            print(f"    ⚠️  Blob/salt pattern not found in puzzle using standard patterns")
            print(f"    🔍 Debug: Searching for hex patterns in raw HTML...")
            blob_in_raw = re.search(r'blob[^a-fA-F0-9]*([a-fA-F0-9]{6,})', question_text, re.IGNORECASE)
            if blob_in_raw:
                blob = blob_in_raw.group(1).strip()
                print(f"    ✓ Found blob in raw HTML after 'blob' keyword: {blob}")
        
        if blob:
            print(f"    📦 Found blob/salt: {blob}")

            # Determine key: Try to fetch /demo2 page (same base domain) to obtain the 8-digit key
            key_str = None
            try:
                if current_url:
                    print(f"    ℹ️  current_url provided: {current_url}")
                    base = '/'.join(current_url.split('/')[:3])
                    print(f"    ℹ️  Extracted base domain: {base}")
                    demo2_path = None
                    # Look for explicit /demo2 mention in the content
                    m = re.search(r'(/demo2\b[^\s\"]*)', full_context)
                    if m:
                        demo2_path = m.group(1)
                        print(f"    ℹ️  Found explicit /demo2 path in content: {demo2_path}")
                    else:
                        demo2_path = '/demo2'
                        print(f"    ℹ️  Using default /demo2 path")

                    demo2_url = base + demo2_path if demo2_path.startswith('/') else base + '/' + demo2_path
                    print(f"    🔎 Attempting to fetch demo2 URL for key: {demo2_url}")
                    demo2_html = fetch_quiz_page(demo2_url)
                    print(f"    ℹ️  demo2_html fetched, length: {len(demo2_html)}")

                    # Try to find an explicit 8-digit key in demo2_html
                    kmatch = re.search(r'\b(\d{8})\b', demo2_html)
                    if kmatch:
                        key_str = kmatch.group(1)
                        print(f"    ✅ Extracted 8-digit key from /demo2 page: {key_str}")
                    else:
                        print(f"    ℹ️  No 8-digit key found in /demo2 HTML, trying formula approach...")
                        # Fall back to emailNumber formula approach if JS patterns present
                        dd_lower = demo2_html.lower()
                        if 'emailnumber' in dd_lower:
                            print(f"    ℹ️  emailnumber pattern found in /demo2 HTML")
                            import hashlib
                            sha1_hash = hashlib.sha1(EMAIL.encode()).hexdigest()
                            email_number = int(sha1_hash[:4], 16)
                            multiplier = 7919
                            offset = 12345
                            modulo = int(1e8)
                            mult_pattern = re.search(r'emailnumber\s*\*\s*(\d+)', dd_lower)
                            if mult_pattern:
                                multiplier = int(mult_pattern.group(1))
                            offset_pattern = re.search(r'\+\s*(\d+)\s*\)', dd_lower)
                            if offset_pattern:
                                offset = int(offset_pattern.group(1))
                            key = (email_number * multiplier + offset) % modulo
                            key_str = str(key).zfill(8)
                            print(f"    ✅ Computed key from demo2 JS formula: {key_str}")
                        else:
                            print(f"    ⚠️  Could not extract key from /demo2 page (no emailnumber pattern)")
                else:
                    print(f"    ⚠️  current_url NOT provided, cannot fetch /demo2")
            except Exception as e:
                print(f"    ⚠️  Error fetching/parsing /demo2: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

            # If we still don't have a key, attempt a local email-based fallback
            if not key_str:
                print(f"    ℹ️  Attempting local email-based fallback...")
                try:
                    import hashlib
                    sha1_hash = hashlib.sha1(EMAIL.encode()).hexdigest()
                    email_number = int(sha1_hash[:4], 16)
                    multiplier = 7919
                    offset = 12345
                    modulo = int(1e8)
                    mult_pattern = re.search(r'emailnumber\s*\*\s*(\d+)', full_context_lower)
                    if mult_pattern:
                        multiplier = int(mult_pattern.group(1))
                    offset_pattern = re.search(r'\+\s*(\d+)\s*\)', full_context_lower)
                    if offset_pattern:
                        offset = int(offset_pattern.group(1))
                    key = (email_number * multiplier + offset) % modulo
                    key_str = str(key).zfill(8)
                    print(f"    🔁 Fallback computed key from context: {key_str}")
                except Exception as e:
                    print(f"    ⚠️  Fallback failed: {type(e).__name__}: {e}")
                    key_str = None

            if not key_str:
                print(f"    ✗ Could not determine 8-digit key required for checksum step")
                return None
            
            print(f"    ℹ️  Key determined: {key_str}")


            # Compute SHA256(key + blob)
            import hashlib
            combined = key_str + blob
            print(f"    🔗 Combined string: {combined}")
            sha256_hash = hashlib.sha256(combined.encode()).hexdigest()
            print(f"    🔐 SHA256 hash: {sha256_hash}")

            # Determine how many characters to return (default 12)
            char_count = 12
            char_pattern = re.search(r'first\s+(\d+)\s+(?:hex\s+)?char', full_context_lower)
            if char_pattern:
                char_count = int(char_pattern.group(1))
                print(f"    📏 Returning first {char_count} characters")

            result = sha256_hash[:char_count]
            print(f"    ✅ Checksum result: {result}")
            return result
    
    # PRIORITY 2: Check for CSV data analysis questions - look for cutoff patterns
    print(f"  🔎 Stage 2: Checking for cutoff values...")
    # Try explicit 'cutoff:' label first (accept commas in numbers)
    cutoff_match = re.search(r'cutoff[:\s]*[\n\s]*([0-9,]+)', full_context, re.IGNORECASE | re.MULTILINE)

    cutoff: Optional[int] = None
    cutoff_operator: Optional[str] = None

    if cutoff_match:
        try:
            cutoff = int(cutoff_match.group(1).replace(',', ''))
            print(f"    ✓ Found cutoff (explicit): {cutoff}")
            ANALYSIS_DEBUG['parsed_cutoff'] = {'operator': None, 'value': cutoff}
        except Exception:
            cutoff = None
            print(f"    ⚠️  Found cutoff token but could not parse number: {cutoff_match.group(1)}")
    else:
        # Try to extract operator+value from natural language (audio transcriptions often say 'greater than 25514')
        op, val = extract_cutoff_operator_and_value(full_context_lower)
        if val is not None:
            cutoff_operator, cutoff = op, val
            # If the spoken transcription contains a phrase like 'greater than or equal' but
            # the numeric match came from a nearby symbol (e.g. '> 25841'), prefer the spoken
            # phrase since audio instructions are authoritative.
            if ('greater than or equal' in full_context_lower or 'at least' in full_context_lower) and cutoff_operator == '>':
                cutoff_operator = '>='
            if ('less than or equal' in full_context_lower or 'at most' in full_context_lower) and cutoff_operator == '<':
                cutoff_operator = '<='

            print(f"    ✓ Extracted cutoff from text: operator='{cutoff_operator}', value={cutoff}")
            ANALYSIS_DEBUG['parsed_cutoff'] = {'operator': cutoff_operator, 'value': cutoff}
        else:
            print(f"    ⊗ No cutoff found in question")
    
    # PRIORITY 3: Look for secret codes in downloaded/scraped files (HIGH PRIORITY)
    print(f"  🔎 Stage 3: Searching for secret codes in scraped data...")
    for filename, content in downloaded_files.items():
        if not filename.endswith('_dataframe') and not filename.endswith('_image') and isinstance(content, str):
            # Skip error messages from failed transcriptions
            if content.startswith('[Audio transcription failed'):
                continue
            
            # Only search if content mentions secret/code or is short text (likely to be answer)
            if 'secret' in content.lower() or 'code' in content.lower() or len(content) < 200:
                print(f"    🔍 Searching for secret in {filename}: {content[:100]}")
                
                # Look for "Secret code is XXXXX" pattern (numbers)
                secret_num_pattern = re.search(r'secret\s+code\s+is\s+([0-9]+)', content, re.IGNORECASE)
                if secret_num_pattern:
                    result = int(secret_num_pattern.group(1))
                    print(f"    ✅ Found secret code (number) in scraped file: {result}")
                    return result
                
                # Look for numbers directly after "secret" or "code"
                code_num_pattern = re.search(r'(?:secret|code)\s*[:\s]*([0-9]+)', content, re.IGNORECASE)
                if code_num_pattern:
                    result = int(code_num_pattern.group(1))
                    print(f"    ✅ Found code number in scraped file: {result}")
                    return result
                
                # Fallback: look for any alphanumeric code
                generic_pattern = re.search(r'(?:secret|code)\s*(?:is|:)?\s*([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
                if generic_pattern:
                    result = generic_pattern.group(1)
                    print(f"    ✅ Found code text in scraped file: {result}")
                    return result
    
    # PRIORITY 4: If we found an answer in the question and no scraped data overrode it, use it
    if potential_answer_from_question:
        print(f"  ✅ Using answer from question: {potential_answer_from_question}")
        return potential_answer_from_question
    
    # PRIORITY 5: Process dataframes for calculations (SUM/COUNT operations)
    print(f"  🔎 Stage 4: Processing dataframes for calculations...")
    for filename, content in downloaded_files.items():
        if filename.endswith('_dataframe'):
            df = content
            
            # Use the first column as the data column (robust to numeric headers)
            if len(df.columns) == 0:
                print(f"    ⚠️  No columns found in dataframe")
                continue

            col_label = str(df.columns[0])
            print(f"    📊 Processing dataframe: {filename.replace('_dataframe', '')} (column: {col_label})")
            # Work on a Series extracted by position to avoid column-name issues
            col_series = df.iloc[:, 0].copy()
            print(f"    📊 DataFrame info: {len(df)} rows, dtypes: {col_series.dtype}")
            try:
                print(f"    📊 Total sum BEFORE filtering: {int(pd.to_numeric(col_series, errors='coerce').sum())}")
            except Exception:
                print(f"    📊 Total sum BEFORE filtering: (could not compute)")
            try:
                print(f"    📊 Sample values: {pd.to_numeric(col_series, errors='coerce').head(5).tolist()}")
            except Exception:
                print(f"    📊 Sample values: (could not compute)")

            # Ensure numeric data in working series
            original_count = len(df)
            col_series = pd.to_numeric(col_series, errors='coerce')
            null_count = int(col_series.isnull().sum())
            if null_count > 0:
                print(f"    ⚠️  Found {null_count} null/non-numeric values, dropping them")
                mask = col_series.notnull()
                df = df[mask]
                col_series = col_series[mask]
                print(f"    📊 Rows after cleaning: {len(df)} (dropped {original_count - len(df)} rows)")
            
            # Check if question implies sum (has cutoff + CSV = likely asking for sum)
            if 'sum' in full_context_lower or 'add' in full_context_lower or (cutoff is not None and 'csv' in full_context_lower):
                print(f"    🧮 Detected SUM operation request")
                if cutoff is not None:
                    # Check if audio transcription was available
                    if not audio_transcription_available:
                        # Calculate both possibilities since we don't have audio instructions
                        sum_above = int(col_series[col_series > cutoff].sum())
                        sum_below = int(col_series[col_series <= cutoff].sum())
                        print(f"      ⚠️  Audio transcription unavailable, calculated both:")
                        print(f"      Sum of values > {cutoff}: {sum_above}")
                        print(f"      Sum of values <= {cutoff}: {sum_below}")

                        # Check context for hints about which operation
                        if 'below' in full_context_lower or 'less' in full_context_lower or 'under' in full_context_lower or '<=' in full_context_lower:
                            result = sum_below
                            print(f"      ✅ Context suggests 'below/less', using sum <= {cutoff}: {result}")
                        elif 'above' in full_context_lower or 'greater' in full_context_lower or 'over' in full_context_lower or '>' in full_context_lower:
                            result = sum_above
                            print(f"      ✅ Context suggests 'above/greater', using sum > {cutoff}: {result}")
                        else:
                            # Default to below when cutoff is mentioned without clear direction
                            result = sum_below
                            print(f"      ✅ No clear context, defaulting to sum <= {cutoff}: {result}")
                        return result
                    else:
                        # Audio transcription available - prefer explicit operator if parsed
                        print(f"      🎧 Analyzing audio instructions for operator...")

                        # Calculate possibilities
                        total_sum = int(col_series.sum())
                        total_count = len(col_series)

                        sum_gte = int(col_series[col_series >= cutoff].sum())
                        sum_gt = int(col_series[col_series > cutoff].sum())
                        count_gte = len(col_series[col_series >= cutoff])
                        count_gt = len(col_series[col_series > cutoff])

                        print(f"      📊 DIAGNOSTIC - All possible answers:")
                        print(f"         Total sum (NO filter): {total_sum} ({total_count} values)")
                        print(f"         Sum >= {cutoff}: {sum_gte} ({count_gte} values)")
                        print(f"         Sum > {cutoff}: {sum_gt} ({count_gt} values)")
                        print(f"         Sum < {cutoff}: {total_sum - sum_gte} ({total_count - count_gte} values)")
                        print(f"         Values AT cutoff (={cutoff}): {int((col_series == cutoff).sum())}")
                        print(f"         Min value in filtered >= : {col_series[col_series >= cutoff].min() if count_gte > 0 else 'N/A'}")

                        # Use parsed operator when available
                        # If parser found an operator, use it decisively
                        if cutoff_operator is not None:
                            if cutoff_operator == '>=':
                                result = sum_gte
                                print(f"      ✅ Parsed operator >=, using sum >= {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '>=', 'cutoff': cutoff, 'rows': count_gte, 'sum': result}
                                return result
                            elif cutoff_operator == '>':
                                result = sum_gt
                                print(f"      ✅ Parsed operator >, using sum > {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '>', 'cutoff': cutoff, 'rows': count_gt, 'sum': result}
                                return result
                            elif cutoff_operator == '<=':
                                result = int(col_series[col_series <= cutoff].sum())
                                print(f"      ✅ Parsed operator <=, using sum <= {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '<=', 'cutoff': cutoff, 'rows': len(col_series[col_series <= cutoff]), 'sum': result}
                                return result
                            elif cutoff_operator == '<':
                                result = int(col_series[col_series < cutoff].sum())
                                print(f"      ✅ Parsed operator <, using sum < {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '<', 'cutoff': cutoff, 'rows': len(col_series[col_series < cutoff]), 'sum': result}
                                return result

                        # Fallback: try to infer operator from textual context
                        if ('greater than or equal' in full_context_lower or 'at least' in full_context_lower or '>=' in full_context_lower):
                            result = sum_gte
                            print(f"      ✅ Inferred operator >= from text, using sum >= {cutoff}: {result}")
                            return result
                        elif ('less than or equal' in full_context_lower or '<=' in full_context_lower or 'at most' in full_context_lower):
                            result = int(col_series[col_series <= cutoff].sum())
                            print(f"      ✅ Inferred operator <= from text, using sum <= {cutoff}: {result}")
                            return result
                        elif 'below' in full_context_lower or ('less than' in full_context_lower and 'equal' not in full_context_lower):
                            result = int(col_series[col_series < cutoff].sum())
                            print(f"      ✅ Inferred operator < from text, using sum < {cutoff}: {result}")
                            return result
                        elif 'above' in full_context_lower or ('greater than' in full_context_lower and 'equal' not in full_context_lower):
                            result = sum_gt
                            print(f"      ✅ Inferred operator > from text, using sum > {cutoff}: {result}")
                            return result
                        else:
                            # Default to > when unsure
                            result = sum_gt
                            print(f"      ⚠️  No operator found, defaulting to sum > {cutoff}: {result}")
                            return result
                else:
                    result = int(col_series.sum())
                    print(f"      ✅ Calculated total sum: {result}")
                    return result
            
            if 'count' in full_context_lower:
                print(f"    🧮 Detected COUNT operation request")
                if cutoff is not None:
                    # Check for exact operator in context when audio is available
                    if audio_transcription_available:
                        print(f"      🎧 Analyzing audio instructions for count operator...")
                        # Prefer parsed operator if available
                        if cutoff_operator is not None:
                            if cutoff_operator == '>=':
                                result = int((col_series >= cutoff).sum())
                                print(f"      ✅ Parsed operator >=, count >= {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '>=', 'cutoff': cutoff, 'rows': result}
                                return result
                            elif cutoff_operator == '>':
                                result = int((col_series > cutoff).sum())
                                print(f"      ✅ Parsed operator >, count > {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '>', 'cutoff': cutoff, 'rows': result}
                                return result
                            elif cutoff_operator == '<=':
                                result = int((col_series <= cutoff).sum())
                                print(f"      ✅ Parsed operator <=, count <= {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '<=', 'cutoff': cutoff, 'rows': result}
                                return result
                            elif cutoff_operator == '<':
                                result = int((col_series < cutoff).sum())
                                print(f"      ✅ Parsed operator <, count < {cutoff}: {result}")
                                ANALYSIS_DEBUG['final_choice'] = {'op': '<', 'cutoff': cutoff, 'rows': result}
                                return result

                        # Fallback: infer from text
                        if ('greater than or equal' in full_context_lower or 'at least' in full_context_lower or '>=' in full_context_lower):
                            result = int((col_series >= cutoff).sum())
                            print(f"      ✅ Inferred operator >= from text, count >= {cutoff}: {result}")
                            return result
                        elif ('less than or equal' in full_context_lower or '<=' in full_context_lower or 'at most' in full_context_lower):
                            result = int((col_series <= cutoff).sum())
                            print(f"      ✅ Inferred operator <= from text, count <= {cutoff}: {result}")
                            return result
                        elif 'below' in full_context_lower or ('less than' in full_context_lower and 'equal' not in full_context_lower):
                            result = int((col_series < cutoff).sum())
                            print(f"      ✅ Inferred operator < from text, count < {cutoff}: {result}")
                            return result
                        else:
                            result = int((col_series > cutoff).sum())
                            print(f"      ✅ Defaulting to count > {cutoff}: {result}")
                            return result
                    else:
                        result = int((col_series > cutoff).sum())
                        print(f"      ✅ Calculated count of values > {cutoff}: {result}")
                        return result
                else:
                    result = int(len(df))
                    print(f"      ✅ Calculated total count: {result}")
                    return result
    
    # PRIORITY 6: Check for visualization requests
    print(f"  🔎 Stage 5: Checking for visualization requests...")
    if any(word in full_context_lower for word in ['chart', 'graph', 'plot', 'visualiz', 'image']):
        for filename, content in downloaded_files.items():
            if filename.endswith('_dataframe'):
                df = content
                print(f"    📊 Visualization requested - checking question for chart type...")
                
                # Detect chart type
                if 'bar' in full_context_lower:
                    viz = create_visualization(df, 'bar')
                    print(f"    ✅ Created bar chart")
                    return viz
                elif 'line' in full_context_lower:
                    viz = create_visualization(df, 'line')
                    print(f"    ✅ Created line chart")
                    return viz
                elif 'scatter' in full_context_lower:
                    viz = create_visualization(df, 'scatter')
                    print(f"    ✅ Created scatter plot")
                    return viz
                elif 'heatmap' in full_context_lower:
                    viz = create_visualization(df, 'heatmap')
                    print(f"    ✅ Created heatmap")
                    return viz
    
    # PRIORITY 7: Check for image analysis requests
    print(f"  🔎 Stage 6: Checking for image analysis requests...")
    if any(word in full_context_lower for word in ['image', 'picture', 'photo', 'vision']):
        for filename, content in downloaded_files.items():
            if filename.endswith('_image'):
                print(f"    🖼️  Analyzing image: {filename}")
                analysis = analyze_image(content, question_text)
                print(f"    ✅ Image analysis: {analysis[:100]}")
                return analysis
    
    print(f"  ⚠️  Python preprocessing could not determine answer - all stages exhausted")
    return None

@app.post("/quiz")
async def handle_quiz(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    try:
        quiz_req = QuizRequest(**body)
    except ValidationError:
        raise HTTPException(status_code=400, detail="Invalid request format")
    
    if quiz_req.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    start_time = time.time()
    current_url = quiz_req.url
    quiz_count = 0
    
    while time.time() - start_time < 170:
        try:
            quiz_count += 1
            print(f"\n=== Quiz {quiz_count} ===")
            print(f"Fetching quiz from: {current_url}")
            
            submit_url, answer = process_quiz(current_url)
            
            # Skip submission if we couldn't determine an answer
            if answer is None:
                print(f"❌ Could not determine answer for quiz {quiz_count}")
                return {"status": "error", "message": f"Could not solve quiz at {current_url}"}
            
            submission = {
                "email": EMAIL,
                "secret": SECRET,
                "url": current_url,
                "answer": answer
            }
            
            print(f"Submitting to: {submit_url}")
            print(f"Submission: {json.dumps(submission, indent=2)}")
            
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(submit_url, json=submission)
                print(f"Response Status: {response.status_code}")
                print(f"Response Text: {response.text[:500]}")
                
                # Check if response is valid JSON
                if not response.text.strip():
                    return {"status": "error", "message": f"Empty response from {submit_url}. Status: {response.status_code}"}
                
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    print(f"Full Response: {response.text}")
                    return {"status": "error", "message": f"Invalid JSON response: {response.text[:200]}"}
                
                print(f"Parsed Result: {result}")
            
            if result.get("correct"):
                print("✅ Answer correct!")
                if "url" in result and result["url"]:
                    current_url = result["url"]
                    print(f"Moving to next quiz: {current_url}")
                    continue
                else:
                    print("🎉 All quizzes completed!")
                    return {"status": "completed", "message": f"All {quiz_count} quizzes solved!", "quizzes_solved": quiz_count}
            else:
                print(f"❌ Answer incorrect. Reason: {result.get('reason', 'No reason provided')}")
                if "url" in result and result["url"]:
                    # Move to next quiz even if answer was wrong (per spec: "you may receive the next url to proceed")
                    current_url = result["url"]
                    print(f"Moving to next quiz: {current_url}")
                    continue
                else:
                    # No next URL provided, quiz sequence ends
                    print("No more quizzes. Ending.")
                    return {"status": "incomplete", "message": f"Completed {quiz_count} quizzes. Last answer was incorrect.", "quizzes_solved": quiz_count}
                    
        except Exception as e:
            print(f"Exception: {e}")
            return {"status": "error", "message": str(e)}
    
    return {"status": "completed"}

@app.get("/")
async def root():
    return {"status": "LLM Quiz API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
