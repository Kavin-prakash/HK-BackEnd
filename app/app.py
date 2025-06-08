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



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langchain_ollama import OllamaLLM
# from langchain.prompts import ChatPromptTemplate
# import pytesseract
# from pdf2image import convert_from_path
# from PIL import Image
# import tempfile
# import os
# import google.generativeai as genai # Import the Gemini library

# app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])  # Update with your React app URL

# # Initialize Ollama model
# model_ollama = OllamaLLM(model="llama3:latest")

# # Configure Gemini API (replace with your actual API key)
# # It's recommended to store your API key securely, e.g., in environment variables.
# # GEMINI_API_KEY = os.getenv("")
# # if not GEMINI_API_KEY:
# #     raise ValueError("GEMINI_API_KEY environment variable not set.")
# genai.configure(api_key="")

# # Initialize Gemini model
# # You might need to specify a model name, e.g., "gemini-pro"
# model_gemini = genai.GenerativeModel('gemini-1.5-flash')
# # model_gemini = genai.GenerativeModel('gemini-2.5-flash')
# # model_gemini = genai.GenerativeModel('gemini-pro')

# # Tesseract and Poppler paths (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'D:\MSCI\OCR\tesseract\tesseract.exe'
# poppler_path = r'D:\MSCI\OCR\poppler-24.08.0\Library\bin'

# # Prompt template
# # This template will be used for both models, but you could define separate ones if needed.
# prompt_template = """
# You are a financial document analysis AI.

# Your task is to extract the following key KPIs from a Capital Call Statement:
# Extract and return only the exact values based on the user's request.
# If no relevant data is found, respond with 'No data found.' Do not provide any additional information beyond the requested values.
# Ensure responses are strictly based on the user's input without relying on pre-trained data or external assumptions.
# Ensure the each and every response that should be in the english language
# ### Phrase Mappings:
# commitment_date:
# - Commitment Date, Date of Commitment, Subscription Date, Closing Date, Commitment Effective Date, Investor Commitment Date

# effective_date:
# - Effective Date, Capital Call Date, Drawdown Date, Notice Date, Issuance Date, Date of Call, Call Date

# capital_call_amount:
# - Capital Call Amount,Total Amount, Drawdown Amount, Amount Called, Amount Due, Capital Requested, Capital Contribution, Amount Payable, This Call Amount, Current Call Amount

# currency:
# - "$" → "USD"
# - "£" → "GBP"
# - "€" → "EUR"

# ### Important Instructions:
# - Use financial logic, context, and proximity to match values with their labels, even if labels differ.
# - Extract even when labels are missing — rely on financial and temporal cues.
# - Do not extract irrelevant or cumulative values (e.g., total to date, historical data, or fees).
# - If a KPI cannot be reliably extracted, return it as null.
# - All numeric values must be normalized as float with dot decimal notation.
# - All dates must be returned in YYYY-MM-DD format.
# - Every JSOn response that should be english language only
# - Only output a valid, minified JSON object without explanations or commentary.

# ### Date Disambiguation Rules:
# - Prefer dates near capital amounts or phrases like "Drawdown", "Call", or "Notice".
# - Use temporal order: commitment_date < effective_date
# - If a date lacks a clear label, infer from its position relative to the capital call text.

# ### Additional Language Handling:
# - If the uploaded PDF is in Spanish, automatically translate the document into English.
# - Extract relevant financial data from the translated English version.
# - Ensure accuracy by using financial logic, context, and proximity rules post-translation.
# - Maintain the output format strictly in JSON, as defined earlier.
# - Every response that should be english language only

# ### Output Format:
# - If the user requests a specific list of data, only return that data.
# - Return JSON response that should be english language only
# - Return should only the user expected data only no extra data it ivoltes the user data
# - Response must be a valid JSON object with no extra text.
# - Every Response should be in the JSON format only
# {text}
# """

# def extract_text_from_file(file_path, file_ext):
#     text = ""
#     if file_ext.lower() == '.pdf':
#         pages = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
#         for i, page in enumerate(pages):
#             text += f"\n--- Page {i+1} ---\n{pytesseract.image_to_string(page)}"
#     elif file_ext.lower() in ['.png', '.jpg', '.jpeg']:
#         image = Image.open(file_path)
#         text = pytesseract.image_to_string(image)
#     else:
#         raise ValueError("Unsupported file type")
#     return text

# @app.route('/upload', methods=['POST'])
# def upload_ollama():
#     userInputPrompt = request.values['userinput']
#     print(f"Check the userInputPrompt for Ollama :{userInputPrompt}")
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     file_ext = os.path.splitext(file.filename)[1]

#     with tempfile.TemporaryDirectory() as tmpdir:
#         file_path = os.path.join(tmpdir, file.filename)
#         file.save(file_path)

#         try:
#             extracted_text = extract_text_from_file(file_path, file_ext)
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

#         # Generate summary using LangChain + Ollama
#         prompt = ChatPromptTemplate.from_template(prompt_template + userInputPrompt)
#         langchain_chain = prompt | model_ollama
#         result = langchain_chain.invoke({"text": extracted_text})

#         print("Ollama Result:", result)
#         return jsonify({"result": result})

# @app.route('/upload_gemini', methods=['POST'])
# def upload_gemini():
#     userInputPrompt = request.values['userinput']
#     print(f"Check the userInputPrompt for Gemini :{userInputPrompt}")
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     file_ext = os.path.splitext(file.filename)[1]

#     with tempfile.TemporaryDirectory() as tmpdir:
#         file_path = os.path.join(tmpdir, file.filename)
#         file.save(file_path)

#         try:
#             extracted_text = extract_text_from_file(file_path, file_ext)
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

#         try:
#             # Prepare the full prompt for Gemini
#             full_prompt = prompt_template.format(text=extracted_text) + userInputPrompt
            
#             # Generate content using the Gemini model
#             response = model_gemini.generate_content(full_prompt)
#             result = response.text

#             print("Gemini Result:", result)
#             return jsonify({"result": result})
#         except Exception as e:
#             return jsonify({'error': f"Error with Gemini model: {str(e)}"}), 500

# @app.route('/')
# def home():
#     return "Upload endpoints are ready at /upload_ollama and /upload_gemini"

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os
import google.generativeai as genai
import json
import fitz # PyMuPDF
import uuid # For generating unique filenames

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Configure paths
pytesseract.pytesseract.tesseract_cmd = r'D:\MSCI\OCR\tesseract\tesseract.exe'
poppler_path = r'D:\MSCI\OCR\poppler-24.08.0\Library\bin'

# Initialize LLM models (as before)
model_ollama = OllamaLLM(model="llama3:latest")
genai.configure(api_key="YOUR_GEMINI_API_KEY") # Replace with your actual Gemini API key
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# Store highlighted PDFs temporarily (in a real app, use a proper storage like S3 or a DB)
# For simplicity, we'll use a dictionary mapping UUIDs to file paths
HIGHLIGHTED_PDF_STORAGE = {}

# Prompt template (unchanged)
prompt_template = """
You are a financial document analysis AI.

Your task is to extract the following key KPIs from a Capital Call Statement:
Extract and return only the exact values based on the user's request.
If no relevant data is found, respond with 'No data found.' Do not provide any additional information beyond the requested values.
Ensure responses are strictly based on the user's input without relying on pre-trained data or external assumptions.
Ensure the each and every response that should be in the english language
### Phrase Mappings:
commitment_date:
- Commitment Date, Date of Commitment, Subscription Date, Closing Date, Commitment Effective Date, Investor Commitment Date

effective_date:
- Effective Date, Capital Call Date, Drawdown Date, Notice Date, Issuance Date, Date of Call, Call Date

capital_call_amount:
- Capital Call Amount,Total Amount, Drawdown Amount, Amount Called, Amount Due, Capital Requested, Capital Contribution, Amount Payable, This Call Amount, Current Call Amount

Balance_Amount:
- Deposit balance remaining , balance , remaining balance , 
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
- All dates must be returned inYYYY-MM-DD format.
- Every JSOn response that should be english language only
- Only output a valid, minified JSON object without explanations or commentary.

### Date Disambiguation Rules:
- Prefer dates near capital amounts or phrases like "Drawdown", "Call", or "Notice".
- Use temporal order: commitment_date < effective_date
- If a date lacks a clear label, infer from its position relative to the capital call text.

### Additional Language Handling:
- If the uploaded PDF is in Spanish, automatically translate the document into English.
- Extract relevant financial data from the translated English version.
- Ensure accuracy by using financial logic, context, and proximity rules post-translation.
- Maintain the output format strictly in JSON, as defined earlier.
- Every response that should be english language only

### Output Format:
- If the user requests a specific list of data, only return that data.
- Return JSON response that should be english language only
- Return should only the user expected data only no extra data it ivoltes the user data
- Response must be a valid JSON object with no extra text.
- Every Response should be in the JSON format only
{text}
"""

def extract_text_with_coords_from_pdf(pdf_path):
    """
    Extracts text and bounding box coordinates from a PDF using Tesseract.
    Returns a list of dictionaries, each containing 'text', 'left', 'top', 'width', 'height', 'page_num'.
    """
    all_text_data = []
    text_content_for_llm = []

    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    for i, page in enumerate(pages):
        page_num = i + 1
        data = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT)

        page_text = []
        for j in range(len(data['text'])):
            text = data['text'][j].strip()
            if text:
                word_info = {
                    'text': text,
                    'left': data['left'][j],
                    'top': data['top'][j],
                    'width': data['width'][j],
                    'height': data['height'][j],
                    'page_num': page_num
                }
                all_text_data.append(word_info)
                page_text.append(text)
        text_content_for_llm.append(f"\n--- Page {page_num} ---\n{' '.join(page_text)}")

    return all_text_data, "\n".join(text_content_for_llm)

def highlight_pdf(original_pdf_path, extracted_data):
    """
    Highlights the extracted data in the PDF using PyMuPDF.
    `extracted_data` is the JSON object from the LLM.
    """
    doc = fitz.open(original_pdf_path)
    # Generate a unique ID for the highlighted PDF
    unique_id = str(uuid.uuid4())
    highlighted_pdf_filename = f"highlighted_{unique_id}.pdf"
    output_pdf_path = os.path.join(tempfile.gettempdir(), highlighted_pdf_filename)

    highlight_color = (1, 1, 0)  # Yellow color (RGB values from 0 to 1)

    for key, value in extracted_data.items():
        if value is None or value == "No data found.":
            continue

        search_value = str(value)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_instances = page.search_for(search_value, quads=True)

            for inst in text_instances:
                rect = inst.rect
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=highlight_color)
                highlight.update()

    doc.save(output_pdf_path)
    doc.close()
    return highlighted_pdf_filename, output_pdf_path

@app.route('/upload', methods=['POST'])
def upload_ollama():
    userInputPrompt = request.values['userinput']
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_ext = os.path.splitext(file.filename)[1]
    if file_ext.lower() not in ['.pdf']:
        return jsonify({'error': 'Only PDF files are supported for highlighting.'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        original_file_path = os.path.join(tmpdir, file.filename)
        file.save(original_file_path)

        try:
            text_coords, extracted_text_for_llm = extract_text_with_coords_from_pdf(original_file_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        prompt = ChatPromptTemplate.from_template(prompt_template + userInputPrompt)
        langchain_chain = prompt | model_ollama
        llm_raw_result = langchain_chain.invoke({"text": extracted_text_for_llm})

        try:
            llm_json_result = json.loads(llm_raw_result)
        except json.JSONDecodeError as e:
            return jsonify({'error': f"LLM output is not valid JSON: {e}", 'raw_output': llm_raw_result}), 500

        try:
            # Get the unique filename and its temporary path
            highlighted_filename, highlighted_temp_path = highlight_pdf(original_file_path, llm_json_result)
            # Store the path in our global dictionary
            HIGHLIGHTED_PDF_STORAGE[highlighted_filename] = highlighted_temp_path

            # Return both the JSON result and the download URL for the PDF
            return jsonify({
                "result": llm_json_result,
                "highlighted_pdf_url": f"/download_pdf/{highlighted_filename}"
            })
        except Exception as e:
            return jsonify({'error': f"Error highlighting PDF: {str(e)}", 'llm_result': llm_json_result}), 500

@app.route('/upload_gemini', methods=['POST'])
def upload_gemini():
    userInputPrompt = request.values['userinput']
    print(f"Print the User Input for Ollama Model",{userInputPrompt})
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_ext = os.path.splitext(file.filename)[1]
    if file_ext.lower() not in ['.pdf']:
        return jsonify({'error': 'Only PDF files are supported for highlighting.'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        original_file_path = os.path.join(tmpdir, file.filename)
        file.save(original_file_path)

        try:
            text_coords, extracted_text_for_llm = extract_text_with_coords_from_pdf(original_file_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        try:
            full_prompt = prompt_template.format(text=extracted_text_for_llm) + userInputPrompt
            response = model_gemini.generate_content(full_prompt)
            llm_raw_result = response.text
            llm_json_result = json.loads(llm_raw_result)
        except json.JSONDecodeError as e:
            return jsonify({'error': f"LLM output is not valid JSON: {e}", 'raw_output': llm_raw_result}), 500
        except Exception as e:
            return jsonify({'error': f"Error with Gemini model: {str(e)}"}), 500

        try:
            highlighted_filename, highlighted_temp_path = highlight_pdf(original_file_path, llm_json_result)
            HIGHLIGHTED_PDF_STORAGE[highlighted_filename] = highlighted_temp_path
            print(llm_json_result,highlighted_filename)
            return jsonify({
                "result": llm_json_result,
                "highlighted_pdf_url": f"/download_pdf/{highlighted_filename}"
            })
        except Exception as e:
            return jsonify({'error': f"Error highlighting PDF: {str(e)}", 'llm_result': llm_json_result}), 500

@app.route('/download_pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    """Serves a highlighted PDF based on its unique filename."""
    pdf_path = HIGHLIGHTED_PDF_STORAGE.get(filename)
    if pdf_path and os.path.exists(pdf_path):
        # We need to remove the file after sending it, or manage cleanup periodically
        # For simplicity, let's keep it for now. In production, use a background task to clean up.
        return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name=filename)
    else:
        return jsonify({'error': 'PDF not found or has expired.'}), 404

@app.route('/')
def home():
    return "Upload endpoints are ready at /upload_ollama and /upload_gemini"

if __name__ == '__main__':
    app.run(debug=True)