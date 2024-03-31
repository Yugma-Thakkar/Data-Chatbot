import os
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer, util
from langchain.chains.question_answering import load_qa_chain

anthropic_secret_key='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'

# sidebar
with st.sidebar:
    st.write("PDF Chatbot")
    st.markdown('''
    ## Instructions
    1. Upload a PDF file
    2. Ask a question
    3. Get the answer
    4. Repeat
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

# def get_embedding(chunk):
#     # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # model = SentenceTransformer('sentence-transformers/sentence-t5-base')
#     embeddings = model.encode(chunk, convert_to_tensor=True)
#     return embeddings

# def index(chunks, name):
#     # embed chunks and save index as a pkl file
#     embeddings = []
#     i = 1
#     for chunk in chunks:
#         embedded_chunk = get_embedding(chunk)
#         embeddings.append(embedded_chunk)
#         print(f"\nEmbedding number: {i}\tshape: {embedded_chunk.shape}\n{embedded_chunk}\n\n")
#         # st.write(f"\nEmbedding number: {i}\tshape: {embedded_chunk.shape}")
#         i += 1
#         dimension = embedded_chunk.shape

#     indx = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
#     indx.add_with_ids(embeddings, range(len(embeddings)))
#     faiss.write_index(indx, f'{name}.pkl')

#     return True

# main content
def main():
    st.title("PDF Chatbot")
    st.write("Upload a PDF file to get started!")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if pdf is not None:
        st.write("PDF uploaded successfully!")
        
        text = extract_text(pdf)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = splitter.split_text(text=text)
        # st.write(f"Number of chunks: {len(chunks)}\n\n")

        store_name = pdf.name[:-4]

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

            st.write("Embeddings indexed successfully!")

        question = st.text_input("Ask a question:")
        
        if question:
            docs = vector_store.similarity_search(query=question, k=3)
            # st.write(docs)

            chat = ChatAnthropic(model_name = "claude-3-haiku-20240307", anthropic_api_key=anthropic_secret_key)
            chain = load_qa_chain(llm=chat, chain_type="stuff")
            response = chain.run(input_documents = docs, question=question)
            st.write(response)

if __name__ == "__main__":
    main()
