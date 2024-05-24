# Document Loading and Analysis
import os

os.environ["OPENAI_API_KEY"] = "..."

from langchain import OpenAI
# Read document, load it and break it in junks
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Measure relatedness of text strings
from langchain.embeddings.openai import OpenAIEmbeddings
# vectore DB, save embeddings for retrival
from langchain.vectorstores import Chroma
# Look up relevant document in ChromaDB and ask relevant question to DB to get answer
from langchain.chains import RetrievalQA
 
loader = TextLoader("./state-of-the-union-23.txt") #file to be uploaded
documents = loader.load() # creates an array of document which is loaded

# split in chunks - if text is too long cant be loaded, break in semantically related chunks
# splits in 3 recursions - \n\n (double new line), \n (new line) and space " "
# chunk overlap = rolling window over paragraph
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# openai ambeddings = vast library of text embeddings, embeddings measure relatedness of text string
# used for searching and clustering
# every embedding is vector of floating point numbers, distance between vector measure the relatedness
# word embedding, sentence embeddings etc. 
# Ex: Cow and chicken are many be close vectors
embeddings = OpenAIEmbeddings()
# use ChromaDB to store embeddings
#Chroma is open source light weight, embedding DB used to store locally 
store = Chroma.from_documents(texts, embeddings, collection_name="state-of-union") # 3 parameters = text, embeddings and coll name to be created

llm = OpenAI(temperature=0)
# retrieve the qa chain from chromaDB
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())
print(chain.run("What did biden talk about Ohio?"))
