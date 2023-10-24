# Multi-RAG: A Multi-Retrieval GPT-4 Prototype

## Summary

This application is designed to provide retrieval-augmented generation (RAG) capabilities on any given document. By utilizing the power of GPT-4 and embeddings from OpenAI, the application can quickly search for and identify relevant passages in a document to generate comprehensive responses to user queries. The application can ingest content from PDF files, vectorize the text for similarity searches, and generate responses that integrate the most pertinent information from the document.

## Components
1. ingest.py
   * Extracts text from PDF documents with sections and page numbers.
   * Uses the pdfminer library to handle PDF content.
   * Embeds text segments using OpenAI's embeddings and stores them in a FAISS vector store for efficient similarity searches.
2. app.py
   * Utilizes streamlit to provide a user-friendly interface for users to input queries.
   * When a query is entered, it:
      * Searches the FAISS vector store to identify the most relevant passages from the ingested document.
      * Generates a comprehensive response that integrates this information using a chain model (LLMChain).
      * Outputs the generated response to the user.

## Dependencies:
For Windows, use cmd terminal
1. Create a new virtual env: python -m venv venv
2. Activate the virtual env: 
   
   Unix: 
   ```
   source venv/bin/activate
   ```
   Windows: 
   ```
   venv\Scripts\activate.bat
   ```
3. Install requirements: 
   
   Unix:
   ```
   ./install_dependencies.sh
   ```
   Windows:
   ```
   install_dependencies.bat
   ```
   Note: replace faiss-cpu with faiss-gpu in the script if you would like to use your GPU. You must have CUDA 11.4 for a successful install. Reference (https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for specific instructions.

Running the app:
1. Set your OpenAI API to a OPENAI_API_KEY var in a .env file in the root directory of the project.
2. Add your document you would like to run retrieval on to the documents directory, and edit the path passed to the documents variable in ingest.py
3. Ingest the PDF with 
```
python ingest.py
```
4. Run the app with 
```
streamlit run app.py
```