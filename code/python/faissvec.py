class FAISSVectorDB:
    def __init__(self,
                 embedding_dimension: int,
                 gpu_id: str = '0',
                 similarity_algo: str = 'cosine'):
        """Initialize FAISS vector database
        Args:
            embedding_dimension: Dimension of embeddings
            gpu_id: GPU device ID
            similarity_algo: 'cosine' or 'l2'
        """
        self.dimension = embedding_dimension
        self.gpu_id = gpu_id
        self.similarity_algo = similarity_algo
        self.index = None
        self.metadata_store = []
        self.document_store = []
        self.use_gpu = False
        self.initialize_index()

    def initialize_index(self):
        """Initialize FAISS index based on similarity algorithm"""
        # Select index type based on similarity algorithm
        if self.similarity_algo == 'cosine':
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
        else:
            self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance

        # Try to move to GPU if available
        if torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, int(self.gpu_id), self.index)
                self.use_gpu = True
            except AttributeError:
                print("GPU FAISS not available, using CPU version")
                self.use_gpu = False

    def add_documents(self,
                     documents: List[str],
                     embeddings: np.ndarray,
                     metadata: List[Dict] = None):
        """Add documents with their embeddings and metadata"""
        if metadata is None:
            metadata = [{} for _ in documents]

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Add vectors to the index
        self.index.add(embeddings.astype(np.float32))
        self.document_store.extend(documents)
        self.metadata_store.extend(metadata)

    def search(self,
              query_embedding: np.ndarray,
              k: int = 1,
              return_embeddings: bool = False) -> List[Dict]:
        """Search for similar documents
        Args:
            query_embedding: Query vector
            k: Number of results to return
            return_embeddings: Whether to return embeddings in results
        """
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search in the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Format results
        results = []
        for dist_array, idx_array in zip(distances, indices):
            result = {
                'distances': dist_array.tolist(),
                'documents': [self.document_store[idx] for idx in idx_array],
                'metadatas': [self.metadata_store[idx] for idx in idx_array],
            }
            if return_embeddings:
                result['embeddings'] = [self.get_embedding(idx) for idx in idx_array]
            results.append(result)

        return results

    def get_embedding(self, idx: int) -> np.ndarray:
        """Get embedding vector for document at index"""
        if not 0 <= idx < len(self.document_store):
            raise ValueError(f"Index {idx} out of range")
        return faiss.vector_to_array(self.index.reconstruct(idx))

class DocumentClassifier:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 device: str = 'cuda',
                 gpu_id: str = '0'):
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.doc_processor = DocumentProcessor()
        self.vectordb = FAISSVectorDB(
            embedding_dimension=self.embedding_model.get_sentence_embedding_dimension(),
            gpu_id=gpu_id
        )



def print_search_results(self, results: List[Dict]):
    print('\n***Results Summary:')
    print(f'Total search docs: {len(results)}')
    print(f'Results per query: {k}')

    # Print each document's results
    for i, doc_results in enumerate(results):
        print(f'\n*** Results for document {i}:')
        
        # Process distances and metadata for this document
        distances = doc_results['distances']
        metadatas = doc_results['metadatas']

        # For each match
        for j, (dist, meta) in enumerate(zip(distances, metadatas)):
            print(f'\nMatch {j+1}:')
            print(f'Label: {meta["label"]}')
            # Convert distance to similarity based on algorithm
            similarity = 1 - dist if self.vectordb.similarity_algo == 'l2' else dist
            print(f'Similarity: {similarity:.4f}')
            
            # Calculate probability using softmax
            exp_distances = np.exp(-np.array(distances))
            probs = exp_distances / exp_distances.sum()
            print(f'Probability: {probs[j]:.4f}')
            
            # Print full metadata
            print(f'Metadata: {meta}')

def get_doc_info():
    print('***get_doc_info()')
    print(f'*Total docs in list: {len(txt_list)}')
    print('*Number of words per document:')
    print(f'  avg_word_cnt:{avg_word_cnt} max_word_cnt:{max_word_cnt} min_word_cnt:{min_word_cnt}')
