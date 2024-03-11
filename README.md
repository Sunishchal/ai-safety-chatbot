# AI Safety Chatbot

This is a simple chatbot that uses retrieval augmented generation (RAG) to answer questions about Blue Dot Impact's AI Safety Fundamentals Course. 
https://aisafetyfundamentals.com/alignment-course-details/

## Setup

Create a virtual environment and install dependencies:


```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a .env file with the following variables:

```plaintext
ANTHROPIC_API_KEY=
PINECONE_API_KEY=
VOYAGE_API_KEY=
```

## Hosting the API server

```python
python main.py
```

## Making requests to the chatbot

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "What is AI alignment?"}' http://localhost:8000/answer
```

## Updating the vector database with new articles

This only needs to be done if new articles become available and it will update the `docs.pkl` file. 


1. Add the new article URLs in the `articles` tuple in the top of the `rag.py` file.
2. Make sure the Beautiful Soup strainer is updated with the relevant div classes containing the text in the new articles. 
3. Run `python rag.py` 
