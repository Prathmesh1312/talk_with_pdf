__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import chainlit as cl
import chromadb
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA 
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# before begin



@cl.on_chat_start
async def on_chat_start():


    # get user's api key
    res = await cl.AskUserMessage(content="Put you Google Generative Language API in input box").send()
    if res:
        google_api_key = res['output']


    # intial setup
    # google_api_key=os.environ.get("GOOGLE_API_KEY")
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key, temperature=0.5)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,)


    # Wait for the user to upload a PDF file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content = "Upload PDF",
            accept=["application/pdf"], 
            max_size_mb=20, 
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()


    # Read the PDF file
    pdf_reader = PdfReader(file.path, 'rb')
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()


    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)
    print(type(texts))


    # create vector database
    vectorstore = Chroma()
    docsearch = vectorstore.from_texts(texts,embeddings)


    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )


    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()


    # Add session variales 
    cl.user_session.set("texts", texts)
    cl.user_session.set("chain", chain)



@cl.on_message
async def on_message(message: str):

    chain = cl.user_session.get("chain") 
    res = await cl.make_async(chain.run)(message.content)

    await cl.Message(res).send()
