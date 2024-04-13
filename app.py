import os
# import langchain
from openai import OpenAI
import anthropic
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

# langchain.debug = True

# EMBEDDING MODELS
# embedding_model = 'sentence-transformers/sentence-t5-base' # okayish, not that good
embedding_model = 'sentence-transformers/msmarco-distilbert-base-dot-prod-v3' # better than the above

# number of chunks to return from the PDF
ret_chunks = 3

with st.sidebar:
    st.write("# PDF Chatbot")

    model_type = st.selectbox(label="Please select your preferred AI model to get started:", placeholder="Select your preferred AI model", options=["OpenAI", 
    "Anthropic"], label_visibility="hidden", index=None)
    model = None
    if model_type == "OpenAI":
        model = st.radio("Select your preferred OpenAI model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=None)
    elif model_type == "Anthropic":
        model = st.radio("Select your preferred Anthropic model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], index=None)
    
    add_vertical_space(1)
    
    api_key = None
    if model is not None and model_type is not None:
        api_key = st.text_input(f"Please enter your {model_type} API Key here:")
        add_vertical_space(1)
        if api_key is not None:
            temperature = st.select_slider("Select the temperature for the model", options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], value=0.0)

    st.divider()
    st.markdown('''
    ## Instructions
    1. Select your preferred AI model
    2. Upload your API key
    3. Upload your PDF file
    4. Ask your question
    5. Get the answer
    6. Repeat!
    ''')
    st.divider()
    add_vertical_space(3)
    st.write("This app was created using [Streamlit](https://streamlit.io/), [Langchain](https://langchain.com/), and [OpenAI](https://openai.com/).")

def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def openai_api_key_test(api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "This is an authentication test. Don't reply with anything other than 'True' or 'False'."},
            {"role": "user", "content": "Return True if you can see this message."},
        ],
        max_tokens=5,
        temperature=0.0
    )
    
    print(f"{model_type} - {response.choices[0].message.content}")
    if response.choices[0].message.content == "True":
        return True
    else:
        return False
    
@st.cache_data
def anthropic_api_key_test(api_key):
    client = anthropic.Client(api_key=api_key)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        system="This is an authentication test. Don't reply with anything other than 'True' or 'False'.",
        messages=[
            {"role": "user", "content": "Return True if you can see this message."} 
        ],
        max_tokens=5,
        temperature=0.0
    )

    print(f"{model_type} - {response.content[0].text}")

    if not response:
        return False

    if response.content[0].text == "True":
        return True
    else:
        return False

def main():
    st.title("PDF Chatbot")

    if model:        

        if api_key:

            if model_type == "OpenAI":
                test_return = openai_api_key_test(api_key)
            elif model_type == "Anthropic":
                test_return = anthropic_api_key_test(api_key)
                
            if test_return == True:
                pdf = st.file_uploader("Please upload a PDF file to get started", type="pdf")

                if pdf is not None:
                    text = extract_text(pdf)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n"]
                    )

                    chunks = splitter.split_text(text=text)
                    store_name = pdf.name[:-4]

                    if os.path.exists(f"{store_name}.pkl"):
                        with open(f"{store_name}.pkl", "rb") as f:
                            vector_store = pickle.load(f)
                    else:
                        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                        with open(f"{store_name}.pkl", "wb") as f:
                            pickle.dump(vector_store, f)

                    question = st.chat_input("Ask a question:")
                    
                    if question is not None:
                        st.chat_message(name="user").write(question)

                        if model_type == "OpenAI":
                            chat = ChatOpenAI(openai_api_key=api_key, model=model, temperature=0.0)
                        elif model_type == "Anthropic":
                            chat = ChatAnthropic(anthropic_api_key=api_key, model=model, temperature=0.0)

                        compressor = LLMChainExtractor.from_llm(chat)
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor = compressor, base_retriever = vector_store.as_retriever(search_type="mmr")
                        )

                        # docs = vector_store.similarity_search(query=question, k=ret_chunks)
                        # docs = vector_store.max_marginal_relevance_search(query=question, k=ret_chunks, fetch_k=10)
                        docs = compression_retriever.get_relevant_documents(query=question)

                        # st.write(f"Top {ret_chunks} most relevant chunks for the question '{question}':")
                        # for i, doc in enumerate(docs):
                        #     st.write(f"{i+1}. {doc}")

                        chain = load_qa_chain(llm=chat, chain_type="stuff")
                        response = chain.run(input_documents=docs, question=question)
                        st.chat_message(name="ai").write(response)  
                    # else:
                    #     st.write("Please ask a question.")

            elif test_return == False:
                st.write("Invalid API key. Please try again.")

            else:
                st.write("Something went wrong. Please try again.")


if __name__ == '__main__':
    main()