import bs4
from itertools import chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import dotenv
import os
import pickle
import voyageai

arciles = (
    "https://epochai.org/blog/compute-trends",
    "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html",
    "https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html",
    "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html",
    "https://aisafetyfundamentals.com/blog/what-is-ai-alignment/?_gl=1*1cj2no2*_ga*MTE5NjYzOTExOC4xNzA2MTE2MTM3*_ga_8W59C8ZY6T*MTcxMDAyMTI0Ni4xNi4xLjE3MTAwMjI5NTUuMC4wLjA.",
    "https://deepmindsafetyresearch.medium.com/goal-misgeneralisation-why-correct-specifications-arent-enough-for-correct-goals-cf96ebc60924",
)

dotenv.load_dotenv('.env')
pc = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))
index_name = "voyage-aisf"
index = pc.Index(index_name)

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content regular-content", "document", "content", "l"))
loader = WebBaseLoader(
    web_paths=articles,
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)
docs = [text_splitter.split_text(doc.page_content) for doc in docs]
docs = [t for t in chain(*docs)]

vo = voyageai.Client()

doc_embds = vo.embed(docs, model="voyage-2", input_type="document").embeddings
vec_list = [{"id": str(k), "values": v} for k, v in enumerate(doc_embds)]
index.upsert(vectors=vec_list)

with open('docs.pkl', 'wb') as f:
    pickle.dump(docs, f)
