from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTAnno, LTChar
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import pickle
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity

#Stores api key into the vectorstore
load_dotenv()

import numpy as np

import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine
import logging  # Step 1: Import logging
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Step 2: Setup the logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def extract_text_with_sections_and_page_numbers(pdf_path):
    extracted_data = []
    section_pattern = re.compile(r'^((\d+\.)+ .+|Appendix [A-Z]—.+)$')
    
    for page_layout in extract_pages(pdf_path):
        section_header = None
        page_text = ''
        page_number = None
        
        for element in page_layout:
            if isinstance(element, (LTTextBox, LTTextLine)):
                text_content = element.get_text().strip()
                
                # Check for section header
                if section_pattern.match(text_content):
                    section_header = text_content
                    continue

                # Assuming page numbers are just numbers and located at the bottom
                if text_content.isdigit() and element.bbox[1] < 100:  # check if y-coordinate is low (near bottom)
                    page_number = int(text_content)
                    continue
                
                page_text += ' ' + text_content
        
        data_entry = {
            'section': section_header,
            'content': page_text.strip(),
            'page_number': page_number
        }

        data_entry = str(data_entry)
        extracted_data.append(data_entry)

    return extracted_data


def ingest_docs():
    """Get documents from a PDF."""
    documents = extract_text_with_sections_and_page_numbers("NIST.SP.800-61r2.pdf")
    logger.info("Finished parsing pdf text")

    embeddings = OpenAIEmbeddings()
    # text = []
    # embeddings = []
    # for doc in documents:
    #     logger.info("Created embedding for: " + str(doc))
    #     embedding = get_embedding(doc, engine="text-embedding-ada-002")
    #     text.append(doc)
    #     embeddings.append(embedding)
        
    # Add to FAISS
    vector_store = FAISS.from_texts(documents, embeddings)
    # Pickle VectorStore
    with open("faiss_vectorstore.pkl", "wb") as f:
        pickle.dump(vector_store, f)

if __name__ == "__main__":
    ingest_docs()