from langchain_community.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(temperature=0)

def generate_answer(llm, query, docs):
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer ONLY from context:\n{context}\n\nQuestion:{query}"
    return llm.predict(prompt)
