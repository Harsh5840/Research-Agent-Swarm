import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import time
import threading
import queue
import gc
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize embeddings model - using local HuggingFace model instead of OpenAI to avoid quota issues
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'max_length': 8192, 'truncation': True}
)

class TimeoutError(Exception):
    pass

def create_vector_store(docs: list[Document], persist_path: str = None, timeout_minutes: int = 60, max_docs: int = None):
    """
    Create a FAISS vector store from LangChain Document objects with improved handling for large datasets.

    Args:
        docs (list[Document]): List of LangChain documents.
        persist_path (str): Optional path to save the FAISS index.
        timeout_minutes (int): Maximum time to spend on embeddings (default: 60 minutes).
        max_docs (int): Maximum number of documents to process (default: None, process all).

    Returns:
        FAISS: The FAISS vector store object.
    """
    # Limit documents if specified
    if max_docs and len(docs) > max_docs:
        print(f"[VECTOR_STORE] Limiting processing to {max_docs} documents (out of {len(docs)} total)")
        docs = docs[:max_docs]
    
    print(f"[VECTOR_STORE] Processing {len(docs)} documents...")
    
    # Truncate documents to prevent context length overflow
    max_chars = 2000  # Increased from 1000 to 2000 for better content retention
    truncated_docs = []
    
    for doc in docs:
        if len(doc.page_content) > max_chars:
            truncated_content = doc.page_content[:max_chars] + "... [truncated]"
            truncated_doc = Document(
                page_content=truncated_content,
                metadata={**doc.metadata, "truncated": True, "original_length": len(doc.page_content)}
            )
            truncated_docs.append(truncated_doc)
        else:
            truncated_docs.append(doc)
    
    print(f"[VECTOR_STORE] Truncated {len(docs) - len(truncated_docs)} documents to fit context length")
    
    # Process documents in batches for better memory management
    batch_size = 50  # Increased from 25 to 50 for better efficiency
    vector_store = None
    
    print(f"[VECTOR_STORE] Creating embeddings in batches of {batch_size}...")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    # Checkpoint file for recovery
    checkpoint_file = None
    if persist_path:
        checkpoint_dir = Path(persist_path).parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / "vector_store_checkpoint.pkl"
    
    # Try to load existing checkpoint
    if checkpoint_file and checkpoint_file.exists():
        try:
            print(f"[VECTOR_STORE] Found checkpoint, attempting to resume...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                vector_store = checkpoint_data['vector_store']
                processed_count = checkpoint_data['processed_count']
                print(f"[VECTOR_STORE] Resumed from checkpoint with {processed_count} documents processed")
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to load checkpoint: {e}")
            vector_store = None
            processed_count = 0
    else:
        processed_count = 0
    
    total_batches = (len(truncated_docs) + batch_size - 1) // batch_size
    
    try:
        for i in range(processed_count, len(truncated_docs), batch_size):
            # Check if we're approaching timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"Vector store creation timed out after {timeout_minutes} minutes")
            
            batch = truncated_docs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"[VECTOR_STORE] Processing batch {batch_num}/{total_batches} ({len(batch)} documents)... [Progress: {i}/{len(truncated_docs)} ({i/len(truncated_docs)*100:.1f}%)]")
            batch_start_time = time.time()
            
            try:
                if vector_store is None:
                    # First batch - create new vector store
                    vector_store = FAISS.from_documents(batch, embeddings_model)
                else:
                    # Subsequent batches - add to existing vector store
                    vector_store.add_documents(batch)
                
                batch_time = time.time() - batch_start_time
                processed_count = i + len(batch)
                print(f"[VECTOR_STORE] Batch {batch_num} completed in {batch_time:.2f}s [Total: {processed_count}/{len(truncated_docs)}]")
                
                # Save checkpoint after each successful batch
                if checkpoint_file:
                    try:
                        checkpoint_data = {
                            'vector_store': vector_store,
                            'processed_count': processed_count,
                            'timestamp': time.time()
                        }
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(checkpoint_data, f)
                    except Exception as e:
                        print(f"[WARN] Failed to save checkpoint: {e}")
                
                # Force garbage collection after each batch to free memory
                gc.collect()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[ERROR] Failed to process batch {batch_num}: {e}")
                # Try to save current progress
                if checkpoint_file and vector_store:
                    try:
                        checkpoint_data = {
                            'vector_store': vector_store,
                            'processed_count': processed_count,
                            'timestamp': time.time()
                        }
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(checkpoint_data, f)
                        print(f"[VECTOR_STORE] Progress saved to checkpoint. You can resume later.")
                    except Exception as save_e:
                        print(f"[WARN] Failed to save checkpoint: {save_e}")
                raise e
        
        total_time = time.time() - start_time
        print(f"[VECTOR_STORE] All {len(truncated_docs)} documents processed successfully in {total_time:.2f}s!")
        
        # Clean up checkpoint file after successful completion
        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"[VECTOR_STORE] Checkpoint file cleaned up")
            except Exception as e:
                print(f"[WARN] Failed to clean up checkpoint: {e}")
        
        if persist_path:
            print(f"[VECTOR_STORE] Saving vector store to {persist_path}...")
            vector_store.save_local(persist_path)
            print(f"[VECTOR_STORE] Vector store saved successfully!")
        
        return vector_store
        
    except Exception as e:
        print(f"[ERROR] Vector store creation failed: {e}")
        if checkpoint_file and checkpoint_file.exists():
            print(f"[INFO] Checkpoint saved at {checkpoint_file}. You can resume the process later.")
        raise e

def load_vector_store(persist_path: str):
    """
    Load a FAISS vector store from disk.

    Args:
        persist_path (str): Path where the FAISS index was saved.

    Returns:
        FAISS: Loaded FAISS vector store object.
    """
    return FAISS.load_local(persist_path, embeddings_model, allow_dangerous_deserialization=True)

def cleanup_checkpoints(persist_path: str):
    """
    Clean up checkpoint files from a previous failed run.
    
    Args:
        persist_path (str): Path where the vector store is/was being saved.
    """
    try:
        checkpoint_dir = Path(persist_path).parent / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_file = checkpoint_dir / "vector_store_checkpoint.pkl"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                print(f"[VECTOR_STORE] Checkpoint file cleaned up")
            if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
                checkpoint_dir.rmdir()
                print(f"[VECTOR_STORE] Empty checkpoint directory removed")
    except Exception as e:
        print(f"[WARN] Failed to clean up checkpoints: {e}")

def similarity_search(vector_store: FAISS, query: str, k: int = 5):
    """
    Perform a similarity search in the vector store.

    Args:
        vector_store (FAISS): The vector store object.
        query (str): Search query.
        k (int): Number of results to return.

    Returns:
        list[Document]: List of most similar documents.
    """
    return vector_store.similarity_search(query, k=k)
