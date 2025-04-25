from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# Optional widget for Colab
try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
except:
    pass

# Set environment variables
os.environ["OPENAI_API_KEY"] = "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"
os.environ["NEO4J_URI"] = "neo4j+s://637263d6.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "R1rI6BDy6GhGvUyue_6XuR10PZ5Up5Ae77mmGflVOoU"

# Base LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

# Neo4j graph instance
graph = Neo4jGraph()

# Vector index setup
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Fulltext index creation
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
)

# Pydantic model for entities
class Entities(BaseModel):
    names: List[str] = Field(..., description="Tourism-related entities from text.")

# Prompt for entity extraction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting Missouri tourism-related entities from the text."),
    ("human", "Extract entities from the following input: {question}"),
])

# Entity extraction chain
entity_chain = prompt | llm.with_structured_output(Entities)

# Full-text search helper

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.
    Allows minor misspellings (~2 character changes).
    """
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""

    query_parts = [f"{word}~2" for word in words]
    return " AND ".join(query_parts)

# Retrieve structured knowledge

def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)}
        )
        result += "\n".join([el['output'] for el in response]) + "\n"
    return result

# Combines structured and unstructured retrieval

def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    return f"Structured data:\n{structured_data}\n\nUnstructured data:\n{'#Document '.join(unstructured_data)}"

# Simple passthrough
_search_query = RunnableLambda(lambda x: x["question"])

# Question answering pipeline

def answerquery(question: str):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel({
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": question})