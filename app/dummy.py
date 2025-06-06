# import pytesseract
# from pdf2image import convert_from_path
# from PIL import Image
# from flask import Flask
# import os
# import re
# import json
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from fastapi import APIRouter, HTTPException
# from fastapi.responses import JSONResponse
# from app.schemas.test_case import TestCaseRequest, TestCaseResponse
# from app.schemas.response import APIResponse
# from app.services.input_validation_services import verify_input
 
# app = Flask(__name__)
# # Initialize Ollama model
# model = OllamaLLM(model="llama3.2")
 
# pdf_path = os.path.join(os.path.dirname(__file__),'asserts', 'Capital-Call-Notice.pdf')
 
# # pdf_path = ('D:\MSCI\Odysseython\Python-0.3\API\asserts\Capital-Call-Notice.pdf')
 
# # Convert pdf to images
# pages = convert_from_path(pdf_path,dpi=300)
 
# # Extract text from images
# text = ""
 
# for page in pages:
#   text = text + pytesseract.image_to_string(page)
#   print(f"The text are :{text}")
#   def generate(text:string)
   
#   template = """
 
#   """
 
#   prompt = ChatPromptTemplate.from_template(template)
#   chain = prompt | model
 
#   result = chain.invoke()
#   print("result",result)
# generate(text)
 
# @app.route('/')
# def home():
#     return "Hello, Flask!"
 
# if __name__ == '__main__':
#     app.run(debug=True)
 
 
 
 
 
 
 
 
   
 