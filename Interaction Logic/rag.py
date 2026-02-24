import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Getting LangChain to log everything
os.environ["LANGSMITH_TRACING"] = "true"

# Getting API key
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

model = init_chat_model(
    "microsoft/Phi-3-mini-4k-instruct",
    model_provider="huggingface",
    temperature=0.7,
    max_tokens=1023,
)

# Initializing embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Storing the embeddings
storage = InMemoryVectorStore(embeddings)

# Loading in a source with consumer information for clothes
# Background information to store for the LLM
bs4_soup = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

fashion_loader = WebBaseLoader(
    web_paths=("https://globusjournal.com/wp-content/uploads/2024/11/GMIT-JD24-161-7-Vibha-Chandrakar.pdf", 
               "https://www.researchgate.net/publication/330769666_Evaluating_fast_fashion_Fast_Fashion_and_Consumer_Behaviour",
               "https://www.researchgate.net/publication/391587135_Consumer's_Buying_Behavior_on_Fashion_Wears",),
               bs_kwargs={"parse_only": bs4_soup}
)
docs = fashion_loader.load()

assert len(docs) == 1

# Splitting docs into chunks
chunk_maker = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
    add_start_index=True, #track index in original document
)

total_chunks = chunk_maker.split_documents(docs)

# Embed and store all the chunks of documents
documents = storage.add_documents(documents=total_chunks)

