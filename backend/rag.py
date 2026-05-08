# rag.py - RAG Pipeline for JengaVest
# Combines ChromaDB vector search + BM25 lexical search for hybrid retrieval

import os
import json
import chromadb
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import tiktoken

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

DOCUMENTS_FOLDER = "./documents"
CHROMA_DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index.json"

CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
COLLECTION_NAME = "financial_documents"


# ─────────────────────────────────────────────────────────
# 1. DOCUMENT INGESTION & TEXT EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extracts text from a PDF file.
    Returns list of tuples: (text, page_number)
    """
    try:
        reader = PdfReader(pdf_path)
        pages = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                pages.append((text, page_num))
        
        return pages
    
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    return len(tokenizer.encode(text))


def chunk_text_recursive(text: str, page_number: int, chunk_size: int = CHUNK_SIZE, 
                        overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Chunks text using a recursive chunking strategy.
    Returns list of chunks with metadata.
    """
    chunks = []
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= chunk_size:
        # If text is smaller than chunk size, return as single chunk
        return [{
            "content": text,
            "page_number": page_number,
            "chunk_index": 0,
            "token_count": len(tokens)
        }]
    
    # Split tokens into chunks with overlap
    chunk_idx = 0
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        
        # Decode tokens back to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        chunks.append({
            "content": chunk_text,
            "page_number": page_number,
            "chunk_index": chunk_idx,
            "token_count": len(chunk_tokens)
        })
        
        # Move start index by (chunk_size - overlap)
        start_idx = end_idx - overlap
        chunk_idx += 1
        
        # Prevent infinite loop if chunk_size <= overlap
        if start_idx >= end_idx:
            break
    
    return chunks


def ingest_documents(documents_folder: str = DOCUMENTS_FOLDER) -> List[Dict]:
    """
    Ingests all PDF documents from the documents folder.
    Returns list of all chunks with metadata.
    """
    all_chunks = []
    
    # Create documents folder if it doesn't exist
    Path(documents_folder).mkdir(exist_ok=True)
    
    pdf_files = list(Path(documents_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {documents_folder}")
        return []
    
    for pdf_path in pdf_files:
        document_name = pdf_path.name
        print(f"Ingesting: {document_name}")
        
        # Extract text from PDF
        pages = extract_text_from_pdf(str(pdf_path))
        
        # Chunk each page
        for page_text, page_num in pages:
            page_chunks = chunk_text_recursive(page_text, page_num)
            
            # Add document name to metadata
            for chunk in page_chunks:
                chunk["document_name"] = document_name
                all_chunks.append(chunk)
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────────────────
# 2. EMBEDDINGS & VECTOR STORAGE (ChromaDB)
# ─────────────────────────────────────────────────────────

def index_chunks_in_chromadb(chunks: List[Dict]) -> None:
    """
    Indexes chunks in ChromaDB for vector search.
    """
    if not chunks:
        print("No chunks to index")
        return
    
    try:
        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append({
                "document_name": chunk["document_name"],
                "page_number": str(chunk["page_number"]),
                "chunk_index": str(chunk["chunk_index"])
            })
        
        # Add to collection
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Indexed {len(chunks)} chunks in ChromaDB")
    
    except Exception as e:
        print(f"Error indexing chunks in ChromaDB: {e}")


def search_chromadb(query: str, n_results: int = 10) -> List[Dict]:
    """
    Searches ChromaDB for relevant chunks.
    Returns list of results with scores.
    """
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert distances to similarity scores (1 - distance for cosine)
        formatted_results = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                score = 1 - results["distances"][0][i]  # Convert distance to similarity
                metadata = results["metadatas"][0][i]
                
                formatted_results.append({
                    "content": doc,
                    "document_name": metadata["document_name"],
                    "page_number": int(metadata["page_number"]),
                    "chunk_index": int(metadata["chunk_index"]),
                    "relevance_score": score,
                    "search_type": "vector"
                })
        
        return formatted_results
    
    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        return []


# ─────────────────────────────────────────────────────────
# 3. BM25 LEXICAL SEARCH
# ─────────────────────────────────────────────────────────

class BM25Index:
    """Manages BM25 indexing and search"""
    
    def __init__(self, index_path: str = BM25_INDEX_PATH):
        self.index_path = index_path
        self.bm25 = None
        self.chunks = []
        self.load_or_create()
    
    def create_index(self, chunks: List[Dict]) -> None:
        """Creates a BM25 index from chunks"""
        self.chunks = chunks
        
        # Tokenize documents
        corpus = [chunk["content"].split() for chunk in chunks]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(corpus)
        
        # Save index to disk
        self.save_index()
    
    def save_index(self) -> None:
        """Saves BM25 index and chunks to disk"""
        try:
            index_data = {
                "chunks": self.chunks,
                "timestamp": datetime.now().isoformat()
            }
            
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            
            print(f"BM25 index saved to {self.index_path}")
        
        except Exception as e:
            print(f"Error saving BM25 index: {e}")
    
    def load_or_create(self) -> None:
        """Loads existing index or creates new one"""
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, "r") as f:
                    index_data = json.load(f)
                    self.chunks = index_data.get("chunks", [])
                
                if self.chunks:
                    corpus = [chunk["content"].split() for chunk in self.chunks]
                    self.bm25 = BM25Okapi(corpus)
                    print(f"Loaded BM25 index with {len(self.chunks)} chunks")
        
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Searches using BM25"""
        if not self.bm25 or not self.chunks:
            return []
        
        try:
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top n results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:n_results]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    chunk = self.chunks[idx]
                    results.append({
                        "content": chunk["content"],
                        "document_name": chunk["document_name"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "relevance_score": float(scores[idx]),
                        "search_type": "lexical"
                    })
            
            return results
        
        except Exception as e:
            print(f"Error searching with BM25: {e}")
            return []


# Global BM25 index instance
bm25_index = BM25Index()


# ─────────────────────────────────────────────────────────
# 4. HYBRID SEARCH & RESULT FUSION
# ─────────────────────────────────────────────────────────

def normalize_scores(results: List[Dict]) -> List[Dict]:
    """Normalizes relevance scores to 0-1 range"""
    if not results:
        return results
    
    max_score = max(r["relevance_score"] for r in results)
    min_score = min(r["relevance_score"] for r in results)
    
    if max_score == min_score:
        for r in results:
            r["relevance_score"] = 1.0
    else:
        for r in results:
            r["relevance_score"] = (r["relevance_score"] - min_score) / (max_score - min_score)
    
    return results


def fuse_results(vector_results: List[Dict], lexical_results: List[Dict]) -> List[Dict]:
    """
    Fuses vector search and lexical search results.
    Uses Reciprocal Rank Fusion (RRF) with weighted combination.
    """
    # Weight vectors more heavily as they capture semantic meaning
    vector_weight = 0.7
    lexical_weight = 0.3
    
    # Normalize scores within each result set
    vector_results = normalize_scores(vector_results)
    lexical_results = normalize_scores(lexical_results)
    
    # Create a combined ranking
    combined = {}
    
    # Add vector results
    for result in vector_results:
        key = (result["document_name"], result["page_number"], 
               result["chunk_index"], result["content"][:50])
        combined[key] = {
            "result": result,
            "score": vector_weight * result["relevance_score"]
        }
    
    # Add/merge lexical results
    for result in lexical_results:
        key = (result["document_name"], result["page_number"], 
               result["chunk_index"], result["content"][:50])
        
        if key in combined:
            combined[key]["score"] += lexical_weight * result["relevance_score"]
        else:
            combined[key] = {
                "result": result,
                "score": lexical_weight * result["relevance_score"]
            }
    
    # Sort by combined score
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    # Return just the results with updated scores
    final_results = []
    for item in sorted_results:
        result = item["result"].copy()
        result["relevance_score"] = item["score"]
        result["search_type"] = "hybrid"
        final_results.append(result)
    
    return final_results


def hybrid_search(query: str, n_results: int = 5) -> List[Dict]:
    """
    Performs hybrid search combining ChromaDB vectors + BM25 lexical.
    Returns top n_results with citations metadata.
    """
    # Get results from both search methods
    vector_results = search_chromadb(query, n_results=20)
    lexical_results = bm25_index.search(query, n_results=20)
    
    # Fuse results
    fused_results = fuse_results(vector_results, lexical_results)
    
    # Return top n results
    return fused_results[:n_results]


# ─────────────────────────────────────────────────────────
# 5. PUBLIC API
# ─────────────────────────────────────────────────────────

def initialize_rag_pipeline() -> None:
    """
    Initializes the RAG pipeline by ingesting documents
    and creating indexes.
    """
    print("Initializing RAG pipeline...")
    
    # Ingest documents
    chunks = ingest_documents()
    
    if chunks:
        # Index in ChromaDB
        index_chunks_in_chromadb(chunks)
        
        # Create BM25 index
        bm25_index.create_index(chunks)
        
        print("RAG pipeline initialized successfully!")
    else:
        print("No documents to index. Please add PDFs to the documents/ folder.")


def search(query: str, n_results: int = 5) -> List[Dict]:
    """
    Main RAG search function.
    Returns relevant chunks with citations.
    """
    if not bm25_index.chunks:
        print("RAG pipeline not initialized. Call initialize_rag_pipeline() first.")
        return []
    
    results = hybrid_search(query, n_results=n_results)
    return results
