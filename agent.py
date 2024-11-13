from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.vector_stores import MetadataFilters,FilterCondition
from llama_index.core.agent import AgentRunner
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import FunctionTool
from llama_index.llms.mistralai import MistralAI
from typing import List,Optional
import os
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from pathlib import Path


db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("multidocument-agent")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

os.environ["MISTRAL_API_KEY"] = "CyX7yanKGxABR4YhSO1fEGjve5m65nZA"
Settings.llm = MistralAI(model="mistral-large-latest")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


# TODO: abstract all of this into a function that takes in a PDF file name
def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    summary_index = SummaryIndex(nodes)

    def vector_query(
        query: str, page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the MetaGPT paper.

        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]

        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts, condition=FilterCondition.OR
            ),
        )
        response = query_engine.query(query)
        return response

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}", fn=vector_query
    )

    def summary_query(
        query: str,
    ) -> str:
        """Perform a summary of document
        query (str): the string query to be embedded.
        """
        summary_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )

        response = summary_engine.query(query)
        return response

    summary_tool = FunctionTool.from_defaults(
        fn=summary_query, name=f"summary_tool_{name}"
    )

    return vector_query_tool, summary_tool

paper_to_tools_dict = {}
papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)
tools = obj_retriever.retrieve("compare and contrast the papers self rag and metagpt")
#
print(tools[0].metadata)
print(tools[1].metadata)

# Create Agent Runner
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], llm=Settings.llm, verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "what are agent roles in MetaGPT, "
    "and then how they communicate with each other."
)