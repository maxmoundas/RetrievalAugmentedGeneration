import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv,find_dotenv
import logging  # Step 1: Import logging
import pickle
from pathlib import Path

# Step 2: Setup the logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


# 1. Vectorise the sales response csv data
logging.info("loading vectorstore")
if not Path("faiss_vectorstore.pkl").exists():
    raise ValueError("faiss_vectorstore.pkl does not exist, please run ingest.py first")
with open("faiss_vectorstore.pkl", "rb") as f:
    global db
    db = pickle.load(f)


# 2. Function for similarity search
def retrieve_info(query):
    logger.info(f"Retrieving info for query: {query}")
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4")

template = """
You are a incident reporter engineer. 
The query you will be asked is an engineer asking for help and you will give me the best answer that 
reflects the question according to your own knowledge and the documentation. The documentation is json data file from a Computer Security Incident Handling Guide. It contains the section header, content, and page_number 

You must follow these rules when constructing your answer:
- when you take information from a source, you must cite it in this format (US, PAGE-NUMBER)
- at the end of the response, you must create an appendix of sources using this citation: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf#page=[PAGE-NUMBER + 9] where PAGE-NUMBER is from the documentation. This link should hyperlink from a (US, PAGE-NUMBER) citation 

Try to output information from multiple sources.

Below is your question that you should answer using the relevant answer:
{message}
 
Here is a list of relevant information from the documentation:
{best_practice}

"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    logger.info(f"Generating response for message: {message}")
    best_practice = retrieve_info(message)
    logger.info(best_practice)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Retrieval Augmented Generation", page_icon=":brain:")

    st.header("Retrieval Augmented Generation :brain:")
    message = st.text_area("Prompt")

    if message:
        logger.info("Generating best practice message in Streamlit app")
        result = generate_response(message)
        st.info(result)

if __name__ == '__main__':
    #python -m streamlit run app.py 
    main()
