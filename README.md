# Multi-RAG: A Multi-Retrieval GPT-4 Prototype

Dependencies:
For Windows, use cmd terminal
1. Create a new virtual env: python -m venv venv
2. Activate the virtual env: 
    Unix:    source venv/bin/activate
    Windows: venv\Scripts\activate.bat
3. Install requirements: 
    Unix:    ./install_dependencies.sh
    Windows: install_dependencies.bat
    Note: replace faiss-cpu with faiss-gpu in the script if you would like to use your GPU. You must have CUDA 11.4 for a successful install. Reference (https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for specific instructions.

Running the app:
1. Set your OpenAI API to a OPENAI_API_KEY var in a .env file in the root directory of the project.
2. Ingest the PDF with 
```
python ingest.py
```
3. Run the app with 
```
streamlit run app.py
```