import os
from src.logging import logger
import mimetypes
from typing import Dict, List, Optional, Tuple, Any
import PyPDF2
import docx
import csv
import json
import pandas as pd



class DocumentExtractor:
    """
    A service for extracting text content from various document types
    stored in the temporary directory
    """
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize the DocumentExtractor
        
        Args:
            temp_dir: Path to temporary directory containing documents
        """
        logger.info("Initializing DocumentExtractor")
        if temp_dir is None:
            self.temp_dir = os.path.join(os.getcwd(), "temp")
        else:
            self.temp_dir = temp_dir
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.info(f"Created temporary directory at: {self.temp_dir}")
        else:
            logger.info(f"Using existing temporary directory at: {self.temp_dir}")
    
    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """
        Extract text from a document file based on its mimetype
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content as a string, or None if extraction failed
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Determine the file type
            mime_type, _ = mimetypes.guess_type(file_path)
            logger.info(f"Extracting text from file: {file_path} (MIME type: {mime_type})")
            
            # Extract text based on file type
            if mime_type == 'application/pdf':
                return self._extract_from_pdf(file_path)
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                return self._extract_from_docx(file_path)
            elif mime_type == 'text/plain':
                return self._extract_from_text(file_path)
            elif mime_type == 'text/csv' or file_path.endswith('.csv'):
                return self._extract_from_csv(file_path)
            elif mime_type == 'application/json' or file_path.endswith('.json'):
                return self._extract_from_json(file_path)
            elif mime_type == 'application/vnd.ms-excel' or mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                return self._extract_from_excel(file_path)
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                # Try to read as plain text for unsupported files
                return self._extract_from_text(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file"""
        logger.info(f"Extracting text from PDF: {file_path}")
        
        extracted_text = ""
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n\n"
        
        logger.info(f"Extracted {len(extracted_text)} characters from PDF")
        return extracted_text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file"""
        logger.info(f"Extracting text from DOCX: {file_path}")
        
        doc = docx.Document(file_path)
        extracted_text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
        
        logger.info(f"Extracted {len(extracted_text)} characters from DOCX")
        return extracted_text
    
    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from a plain text file"""
        logger.info(f"Extracting text from plain text file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as text_file:
            extracted_text = text_file.read()
        
        logger.info(f"Extracted {len(extracted_text)} characters from text file")
        return extracted_text
    
    def _extract_from_csv(self, file_path: str) -> str:
        """Extract text from a CSV file"""
        logger.info(f"Extracting text from CSV: {file_path}")
        
        extracted_text = ""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                extracted_text += ", ".join(row) + "\n"
        
        logger.info(f"Extracted {len(extracted_text)} characters from CSV")
        return extracted_text
    
    def _extract_from_json(self, file_path: str) -> str:
        """Extract text from a JSON file"""
        logger.info(f"Extracting text from JSON: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as json_file:
            data = json.load(json_file)
            # Convert JSON to string representation
            extracted_text = json.dumps(data, indent=2)
        
        logger.info(f"Extracted {len(extracted_text)} characters from JSON")
        return extracted_text
    
    def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        logger.info(f"Extracting text from Excel: {file_path}")
        
        extracted_text = ""
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            extracted_text += f"Sheet: {sheet_name}\n"
            extracted_text += df.to_string(index=False) + "\n\n"
        
        logger.info(f"Extracted {len(extracted_text)} characters from Excel")
        return extracted_text
    
    def extract_from_directory(self, directory_path: str = None) -> Dict[str, str]:
        """
        Extract text from all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents (default: self.temp_dir)
            
        Returns:
            Dictionary mapping filenames to extracted text content
        """
        if directory_path is None:
            directory_path = self.temp_dir
        
        logger.info(f"Extracting text from all documents in: {directory_path}")
        
        extracted_contents = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                extracted_text = self.extract_text_from_file(file_path)
                if extracted_text:
                    extracted_contents[filename] = extracted_text
                    logger.info(f"Extracted text from: {filename}")
                else:
                    logger.warning(f"Failed to extract text from: {filename}")
        
        logger.info(f"Extracted text from {len(extracted_contents)} documents in directory")
        return extracted_contents
    
    def extract_from_document_list(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract text from a list of document dictionaries with local_path field
        
        Args:
            documents: List of document dictionaries with local_path field
            
        Returns:
            List of document dictionaries with added extracted_text field
        """
        logger.info(f"Extracting text from {len(documents)} documents")
        
        for doc in documents:
            if "local_path" in doc and doc["local_path"]:
                extracted_text = self.extract_text_from_file(doc["local_path"])
                if extracted_text:
                    doc["extracted_text"] = extracted_text
                    logger.info(f"Extracted text from document: {doc.get('title', 'Untitled')}")
                else:
                    logger.warning(f"Failed to extract text from document: {doc.get('title', 'Untitled')}")
            else:
                logger.warning(f"Document has no local_path: {doc.get('title', 'Untitled')}")
        
        extraction_count = sum(1 for doc in documents if "extracted_text" in doc)
        logger.info(f"Successfully extracted text from {extraction_count} out of {len(documents)} documents")
        
        return documents 