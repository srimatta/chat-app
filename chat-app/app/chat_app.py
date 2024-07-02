import os
import sys

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from app.documents_ingestion import get_vector_db
from app.re_ranker import get_re_ranker

if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ''

    llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')
    vectordb = get_vector_db()
    retriever = vectordb.as_retriever(search_kwargs={'k': 3})
    compression_retriever = get_re_ranker(retriever)
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=compression_retriever,
        return_source_documents=True,
        verbose=True
    )

    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    chat_history = []
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the DocBot. You are now ready to start interacting with your documents')
    print('---------------------------------------------------------------------------------')
    while True:
        query = input(f"{green}Prompt: ")
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        result = chain.invoke(
            {"question": query, "chat_history": chat_history})
        print(f"{white}Answer: " + result["answer"])
        chat_history.append((query, result["answer"]))
