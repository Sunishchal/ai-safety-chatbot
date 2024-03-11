import os
import voyageai
from pinecone import Pinecone
from anthropic import Anthropic
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import dotenv
import os
import uvicorn

dotenv.load_dotenv('.env')
pc = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))
index_name = "voyage-aisf"
index = pc.Index(index_name)

with open('docs.pkl', 'rb') as f:
    docs = pickle.load(f)

class ClaudeChat:
    def __init__(self, query: str):
        self.claude = Anthropic()
        self.vo = voyageai.Client()
        self.messages = []
        self.query = query
        self.sys_prompt = \
"""You are a Q&A chatbot that answers questions about AI safety topics.
The questions will come with relevant documents at the beginning indicated by XML tags like <doc_1> followed by the question you need to answer.
Use the information in the documents to answer the question if possible.
In case the question can't be answered from the information in the documents, feel free to augment using the knowledge contained in your model."""

    def initial_message(self):
        q = self.vo.embed(self.query, model="voyage-2", input_type="query").embeddings
        ret = index.query(
          vector=q,
          top_k=3,
          include_values=True
        )
        ret_ids = [int(m['id']) for m in ret['matches']]

        question = \
f"""<doc_1>
{docs[ret_ids[0]]}
</doc_1>

<doc_2>
{docs[ret_ids[1]]}
</doc_2>

<doc_3>
{docs[ret_ids[2]]}
</doc_3>

Question:
{self.query}"""

        self.messages = [{"role": "user", "content": question}]

    def claude_chat(self):
        res = self.claude.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.1,
            system=self.sys_prompt,
            messages=self.messages,
        )
        answer = res.content[0].text
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def follow_up(self):
        ask = input()
        self.messages.append({"role": "user", "content": ask})
        return self.claude_chat()

    def run(self):
        self.initial_message()
        initial_response = self.claude_chat()
        print(initial_response)
        while True:
            print(self.follow_up())

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    chat = ClaudeChat(request.query)
    chat.initial_message()
    initial_response = chat.claude_chat()
    return {"response": initial_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
