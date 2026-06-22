import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [
        dir for dir in current_dir.iterdir()
        if dir.is_dir() and dir.name.startswith("chroma")
    ]
    if not chroma_dirs:
        return {}
    # TODO: Loop through each discovered directory
    for dir in chroma_dirs:
        # TODO: Wrap connection attempt in try-except block for error handling
    
            # TODO: Initialize database client with directory path and configuration settings
        try:
            client = chromadb.PersistentClient(path=str(dir))
            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()
            print(f"Found collections in {dir}: {collections}")
            # TODO: Loop through each collection found
            for collection in collections:
                # TODO: Create unique identifier key combining directory and collection names
                collection_key = f"{dir.name}:{collection.name}"
                collection_count = collection.count()
                print(f"collection_key {collection_key}, collection count {collection_count}")
       
                # TODO: Build information dictionary containing:
                backend_info = {
                    "directory": str(dir), # TODO: Store directory path as string
                    "collection_name": collection.name, # TODO: Store collection name
                    "display_name": f"{collection.name} ({dir.name})", # TODO: Create user-friendly display name
                    "document_count": collection_count   # TODO: Get document count with fallback for unsupported operations
                }
                # TODO: Add collection information to backends dictionary
                backends[collection_key] = backend_info
        # TODO: Handle connection or access errors gracefully
        except Exception as e:
            # TODO: Create fallback entry for inaccessible directories
            # TODO: Include error information in display name with truncation
            # TODO: Set appropriate fallback values for missing information
            print(f"Error connecting to {dir}: {e}")
            error_message = str(e)
            backends[dir.name] = {
                "path": str(dir),
                "collection":None,
                "display_name": f"{dir.name} (Error: {error_message[:30]}...)",
                "document_count": 0,
                "error": error_message
            }

    # TODO: Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO: Create a chomadb persistentclient
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        # TODO: Return the collection with the collection_name
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    where_filter = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}
        print(f"Filtering by mission: {mission_filter}")

    # TODO: Execute database query with the following parameters:
    results = collection.query(
        query_texts=[query],  # TODO: Pass search query in the required format
        n_results=n_results, # TODO: Set maximum number of results to return
        where= where_filter  # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    )
    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section
    context_parts = ["\n\n--- Retrieved Documents ---"]

    # TODO: Loop through paired documents and their metadata using enumeration
    for i, (document, metadata) in enumerate(zip(documents, metadatas), 1): # i starts from 1
        # TODO: Extract mission information from metadata with fallback value
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission = metadata.get("mission", "Unknown Mission").replace("_", " ").title()
        # TODO: Extract source information from metadata with fallback value
        source = metadata.get("source", "Unknown Source")
        # TODO: Extract category information from metadata with fallback value
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        category = metadata.get("category", "General").replace("_", " ").title()
    
        # TODO: Create formatted source header with index number and extracted information
        source_header =[
            f"\n Source{i} \n"
            f"Mission: {mission} | "
            f"Category: {category} | "
            f"Source: {source} "
        ] 
        # TODO: Add source header to context parts list
        context_parts.append(source_header)
        # TODO: Check document length and truncate if necessary
        truncated_doc = document[:500] + "..." if len(document) > 500 else document
        # TODO: Add truncated or full document content to context parts list
        context_parts.append(truncated_doc)
    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
