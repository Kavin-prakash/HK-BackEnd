# from langchain_ollama import OllamaLLM
# from langchain.prompts import ChatPromptTemplate
# import pytesseract
# from pdf2image import convert_from_path
# from flask import Flask
# import os

# app = Flask(__name__)

# # Initialize Ollama model
# model = OllamaLLM(model="llama3:latest")

# # Path to the PDF file
# pdf_path = os.path.join(os.path.dirname(__file__), 'asserts', 'Capital-Call-Notice.pdf')

# # Poppler path for Windows
# poppler_path = r'D:\MSCI\OCR\poppler-24.08.0\Library\bin'

# # Set path to Tesseract executable (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r'D:\MSCI\OCR\tesseract\tesseract.exe'

# # Convert PDF to images
# pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

# # Extract text from each page
# text = ""
# for i, page in enumerate(pages):
#     page_text = pytesseract.image_to_string(page)
#     text += f"\n--- Page {i+1} ---\n{page_text}"

# # Save extracted text to a file
# with open(os.path.join(os.path.dirname(__file__), 'extracted_text.txt'), 'w', encoding='utf-8') as f:
#     f.write(text)

# # Function to generate response using LangChain + Ollama
# def generate_summary(text: str):
#     prompt_template = """
  
# You are a financial document analysis AI.

# Your task is to extract the following key KPIs from a Capital Call Statement:

# - commitment_date: Date when the investor commitment was made (in ISO format: YYYY-MM-DD)
# - effective_date: Official date of the capital call or drawdown (in ISO format)
# - lp_name: Name of the Limited Partner or Investor
# - fund_name: Name of the Fund issuing the capital call
# - capital_call_amount: Amount of capital being called in this notice (float, no symbols)
# - currency: The 3-letter currency code (USD, GBP, EUR, etc.)

# ### Important Instructions:
# - Use financial logic, context, and proximity to match values with their labels, even if labels differ.
# - Extract even when labels are missing — rely on financial and temporal cues.
# - Do not extract irrelevant or cumulative values (e.g., total to date, historical data, or fees).
# - If a KPI cannot be reliably extracted, return it as null.
# - All numeric values must be normalized as float with dot decimal notation.
# - All dates must be returned in YYYY-MM-DD format.
# - Only output a valid, minified JSON object without explanations or commentary.

# ### Phrase Mappings:

# commitment_date:
# - Commitment Date, Date of Commitment, Subscription Date, Closing Date, Commitment Effective Date, Investor Commitment Date

# effective_date:
# - Effective Date, Capital Call Date, Drawdown Date, Notice Date, Issuance Date, Date of Call, Call Date

# capital_call_amount:
# - Capital Call Amount, Drawdown Amount, Amount Called, Amount Due, Capital Requested, Capital Contribution, Amount Payable, This Call Amount, Current Call Amount

# currency:
# - "$" → "USD"
# - "£" → "GBP"
# - "€" → "EUR"

# ### Date Disambiguation Rules:
# - Prefer dates near capital amounts or phrases like "Drawdown", "Call", or "Notice".
# - Use temporal order: commitment_date < effective_date
# - If a date lacks a clear label, infer from its position relative to the capital call text.

# ### Output Format:
# Return only valid JSON with keys:
# {{
#   "commitment_date": "YYYY-MM-DD or null",
#   "effective_date": "YYYY-MM-DD or null",
#   "lp_name": "string or null",
#   "fund_name": "string or null",
#   "capital_call_amount": float or null,
#   "currency": "USD/GBP/EUR or null"
# }}

# {text}

#     """
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     langchain_chain = prompt | model
#     result = langchain_chain.invoke({"text": text})
#     print("Summary:\n", result)

# # Generate summary from extracted text
# generate_summary(text)

# @app.route('/')
# def home():
#     return "PDF text extraction and summarization complete. Check the console or 'extracted_text.txt'."

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Update with your React app URL

# Initialize Ollama model
model = OllamaLLM(model="llama3:latest")

# Tesseract and Poppler paths (Windows)
pytesseract.pytesseract.tesseract_cmd = r'D:\MSCI\OCR\tesseract\tesseract.exe'
poppler_path = r'D:\MSCI\OCR\poppler-24.08.0\Library\bin'

# Prompt template
prompt_template = """
You are a financial document analysis AI.

Your task is to extract the following key KPIs from a Capital Call Statement:
Extract and return only the exact values based on the user's request.
If no relevant data is found, respond with 'No data found.' Do not provide any additional information beyond the requested values.
 Ensure responses are strictly based on the user's input without relying on pre-trained data or external assumptions



### Phrase Mappings:

commitment_date:
- Commitment Date, Date of Commitment, Subscription Date, Closing Date, Commitment Effective Date, Investor Commitment Date

effective_date:
- Effective Date, Capital Call Date, Drawdown Date, Notice Date, Issuance Date, Date of Call, Call Date

capital_call_amount:
- Capital Call Amount, Drawdown Amount, Amount Called, Amount Due, Capital Requested, Capital Contribution, Amount Payable, This Call Amount, Current Call Amount

currency:
- "$" → "USD"
- "£" → "GBP"
- "€" → "EUR"
 
### Important Instructions:
- Use financial logic, context, and proximity to match values with their labels, even if labels differ.
- Extract even when labels are missing — rely on financial and temporal cues.
- Do not extract irrelevant or cumulative values (e.g., total to date, historical data, or fees).
- If a KPI cannot be reliably extracted, return it as null.
- All numeric values must be normalized as float with dot decimal notation.
- All dates must be returned in YYYY-MM-DD format.
- Only output a valid, minified JSON object without explanations or commentary.


### Date Disambiguation Rules:
- Prefer dates near capital amounts or phrases like "Drawdown", "Call", or "Notice".
- Use temporal order: commitment_date < effective_date
- If a date lacks a clear label, infer from its position relative to the capital call text.

### Output Format:
Here I have address one thing if the user asks for particular list of data means you should
give that kind of data only 
You should returm only the JSON values alone apart from that I dont need any text in the response

{text}
"""

def extract_text_from_file(file_path, file_ext):
    text = ""
    if file_ext.lower() == '.pdf':
        pages = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
        for i, page in enumerate(pages):
            text += f"\n--- Page {i+1} ---\n{pytesseract.image_to_string(page)}"
    elif file_ext.lower() in ['.png', '.jpg', '.jpeg']:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file type")
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    userInputPrompt = request.values['userinput']
    print(f"Check the userInputPrompt :{userInputPrompt}")
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_ext = os.path.splitext(file.filename)[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        file.save(file_path)

        try:
            extracted_text = extract_text_from_file(file_path, file_ext)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        # Generate summary using LangChain + Ollama
        prompt = ChatPromptTemplate.from_template(prompt_template+userInputPrompt)
        langchain_chain = prompt | model
        result = langchain_chain.invoke({"text": extracted_text})

        print(result)
        return jsonify({"result": result})
#     print("Summary:\n", result)
    

@app.route('/')
def home():
    return "Upload endpoint is ready at /upload"

if __name__ == '__main__':
    app.run(debug=True)
