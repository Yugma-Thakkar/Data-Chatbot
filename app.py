import os
import openai
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# sidebar
with st.sidebar:
    st.markdown('''
    # PDF Chatbot
    ## Instructions
    1. Upload your OpenAI API key
    2. Upload your PDF file
    3. Ask your question
    4. Get the answer
    5. Enjoy!
    ''')

    add_vertical_space(5)
    st.write("This app was created using [Streamlit](https://streamlit.io/), [Langchain](https://langchain.com/), and [OpenAI](https://openai.com/).")

def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# later on, figure out how to test the OpenAI API key without having to run every time
# test.txt -> <API_KEY> = True | False
def api_key_test(api_key):
    # try:
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "system", "content": "This is just a test."},
    #             {"role": "user", "content": "Reply with 'True' if you can see this message."},
    #         ],
    #         max_tokens=5,
    #         api_key=api_key,
    #         temperature=0.0
    #     )
        
    #     print(response.choices[0].message.content)
    #     if response.choices[0].message.content == "True":
    #         # create a file to store the response
    #         with open("test.txt", "w") as f:
    #             f.write(response.choices[0].message.content)
    #         return True

    # except Exception as e:
    #     return Exception
    return True

# main content
def main():
    st.title("PDF Chatbot")
    # st.write("Upload a PDF file to get started!")
    
    openai_api_key = st.text_input("OpenAI API Key")

    if not openai_api_key:
        st.write("Please upload your OpenAI API key to get started.")
    elif openai_api_key:
        # st.write("OpenAI API key uploaded successfully!")
        # st.write(openai_api_key)

        test_return = api_key_test(openai_api_key)

        if test_return == True:

            pdf = st.file_uploader("Choose a PDF file", type="pdf")

            if pdf is None:
                st.write("Please upload a PDF file to get started.")
            elif pdf is not None:
                # st.write("PDF uploaded successfully!")
                
                text = extract_text(pdf)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )

                chunks = splitter.split_text(text=text)
                # st.write(f"Number of chunks: {len(chunks)}\n\n")

                store_name = pdf.name[:-4]
                # st.write(store_name)
                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        vector_store = pickle.load(f)
                        # st.write("Index loaded successfully!")
                else:
                    model = 'sentence-transformers/sentence-t5-base'
                    embeddings = HuggingFaceEmbeddings(model_name=model)
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(vector_store, f)

                    # st.write("Embeddings indexed successfully!")

                question = st.text_input("Ask a question:")

                if question:
                    docs = vector_store.similarity_search(query=question, k=3)

                    chat = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
                    chain = load_qa_chain(llm=chat, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=question)
                        print(cb)
                        st.write(response)
                
        else:
            st.write("Invalid OpenAI API key. Please try again.")

if __name__ == '__main__':
    main()