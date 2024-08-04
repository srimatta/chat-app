import os
import sys

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.documents_ingestion import get_vector_db
from app.re_ranker import get_re_ranker

def pretty_print_docs(docs):
    print("pretty_print_docs:")
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ''

    from langchain.globals import set_debug
    # set_debug(True)
    import phoenix as px
    from phoenix.trace.langchain import LangChainInstrumentor

    session = px.launch_app()
    # Initialize Langchain auto-instrumentation
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini', verbose=True, streaming=True)
    vectordb = get_vector_db(source_docs_folder='/Users/srinivas_work/Desktop/apps/chat-app/chat-app/docs/')
    retriever_ = vectordb.as_retriever(search_kwargs={'k': 3})


    compression_retriever = get_re_ranker(retriever_)

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_
    )


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """
    Answer the question based only on the following context: 
    {context}

    Question: {question}
    """

    # Create the PromptTemplate instance
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    from langchain_core.runnables import RunnableParallel

    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": compression_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

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
        result = rag_chain_with_source.invoke(query)
        print(f"{white}Answer: " ,  result['answer'])
        print(f"{white}Sources: ", result['context'])
