# 🚀 Vector Database Mastery: Zero to Hero Journey

*Welcome to your epic quest to master vector databases! Each quest builds upon the previous one, unlocking new powers along the way.*

---

## 🎯 Learning Philosophy

**Progressive Mastery**: Each exercise introduces exactly 2-3 new concepts while reinforcing previous knowledge.  
**Practical Focus**: Every concept is immediately applied in code.  
**Fun Factor**: Gamified progression with real-world scenarios that don't sacrifice technical depth.

---

## 📚 **Pre-Quest Setup** (30 minutes)
*Your hero needs the right tools before the adventure begins*

### Essential Gear
```bash
# Install your hero toolkit
pip install chromadb sentence-transformers numpy pandas matplotlib
pip install faiss-cpu  # or faiss-gpu if you have CUDA
pip install openai     # for advanced quests
```

### Knowledge Prerequisites
- Python basics (lists, dictionaries, functions)
- Basic understanding of similarity (like recommendations: "people who liked X also liked Y")
- Curiosity about how search engines find relevant results

### 📚 Conceptos Fundamentales

Antes de empezar, asegúrate de entender:
- **Vectores**: Representaciones numéricas de datos (texto, imágenes, audio)
- **Embeddings**: Vectores que capturan el significado semántico
- **Similitud coseno**: Medida para comparar vectores
- **Dimensionalidad**: Número de elementos en un vector

**¿Qué es una Base de Datos Vectorial?**

Imagina que quieres buscar imágenes similares a un "perro golden retriever". Las bases de datos tradicionales buscan texto exacto. Las vectoriales funcionan diferente:

1. **Representación Vectorial**: Un modelo de IA convierte datos complejos en vectores que capturan significado semántico
2. **Búsqueda por Similitud**: Encuentra vectores "cercanos" matemáticamente, no coincidencias exactas

---

## 🎮 **Quest 1: Vector Playground** - *The Awakening*
**Power Level**: Beginner  
**Time**: 2-3 hours  
**New Concepts**: Vectors, Cosine Similarity, Manual Embeddings

### 🎪 The Challenge: "Pet Similarity Detective"
You're building a pet matching service! Dogs should match with dogs, cats with cats, but what about a "friendly dog" vs "loyal pet"?

```python
import numpy as np
import matplotlib.pyplot as plt

# Your first vector spell - represent pets as magical coordinates
pet_vectors = {
    "loyal_dog": [1.0, 0.9, 0.2, 0.8],      # [domestic, loyal, independent, playful]
    "house_cat": [1.0, 0.3, 0.9, 0.6],      # [domestic, loyal, independent, playful]
    "wild_wolf": [0.1, 0.8, 0.9, 0.4],      # [domestic, loyal, independent, playful]
    "goldfish": [1.0, 0.1, 0.1, 0.3],       # [domestic, loyal, independent, playful]
    "gato": [1, 0, 1, 0, 1],                 # animal, doméstico, peludo
    "perro": [1, 0, 1, 0, 1],                # animal, doméstico, peludo  
    "león": [1, 1, 1, 0, 0],                 # animal, salvaje, peludo
    "mesa": [0, 0, 0, 1, 0],                 # objeto, mueble
}

def cosine_similarity_spell(vec_a, vec_b):
    """Cast the similarity spell between two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)

def find_most_similar_pet(query_pet, pet_database):
    """Find the pet most similar to your query"""
    query_vector = pet_database[query_pet]
    similarities = {}
    
    for pet_name, pet_vector in pet_database.items():
        if pet_name != query_pet:
            similarity = cosine_similarity_spell(query_vector, pet_vector)
            similarities[pet_name] = similarity
    
    # Return the most similar pet
    most_similar = max(similarities, key=similarities.get)
    return most_similar, similarities[most_similar]

# 🎯 Your First Quest Mission
print("🔍 Pet Similarity Detective Results:")
similar_pet, score = find_most_similar_pet("loyal_dog", pet_vectors)
print(f"Most similar to loyal_dog: {similar_pet} (similarity: {score:.3f})")

# Challenge: Add 5 more pets and find interesting similarity patterns!
```

### 🏆 Quest Completion Requirements
1. **Bronze**: Run the code and understand why certain pets are similar
2. **Silver**: Add 5 new pets with logical vector values
3. **Gold**: Create a visualization showing pet similarities as a heatmap
4. **Legendary**: Implement different distance metrics (Euclidean, Manhattan) and compare results

### Entregables
- Script que almacene 10 vectores de palabras
- Función de búsqueda que retorne el más similar
- Análisis de por qué ciertos vectores son más similares

---

## 🎮 **Quest 2: First Real Vector Database** - *The Foundation*
**Power Level**: Novice  
**Time**: 3-4 hours  
**New Concepts**: ChromaDB, Real Embeddings, Collections, Metadata

### 🎪 The Challenge: "Movie Recommendation Wizard"
Build a movie recommendation system using real sentence embeddings!

```python
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize your vector database spell
client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create your movie collection
movies_collection = client.create_collection(
    name="epic_movies",
    embedding_function=sentence_transformer_ef
)

# Load your movie database
movie_descriptions = [
    "A space opera about rebels fighting an evil empire with laser swords",
    "A wizard boy learns magic and fights dark forces at a school",
    "Giant robots from space transform into vehicles and fight evil",
    "A young woman with ice powers learns to control her abilities",
    "Superheroes assemble to save the world from alien invasion",
    "A clownfish father searches the ocean for his lost son",
    "Toys come alive when humans aren't watching",
    "A princess falls in love with a beast in an enchanted castle",
    "El gato duerme en el sofá",
    "El perro juega en el parque", 
    "Me gusta programar en Python",
    "Las bases de datos son importantes"
]

movie_titles = [
    "Star Wars", "Harry Potter", "Transformers", "Frozen",
    "Avengers", "Finding Nemo", "Toy Story", "Beauty and Beast",
    "Cat Movie", "Dog Movie", "Python Documentary", "Database Film"
]

# Add movies to your database with metadata
movies_collection.add(
    documents=movie_descriptions,
    metadatas=[{
        "title": title, 
        "genre": "family" if i < 8 else "documentary",
        "language": "English" if i < 8 else "Spanish",
        "category": "entertainment" if i < 8 else "tech"
    } for i, title in enumerate(movie_titles)],
    ids=[f"movie_{i}" for i in range(len(movie_titles))]
)

def recommend_movies(user_preference, num_recommendations=3, filters=None):
    """Your movie recommendation spell with optional filtering"""
    results = movies_collection.query(
        query_texts=[user_preference],
        n_results=num_recommendations,
        where=filters
    )
    
    print(f"🎬 Movies recommended for: '{user_preference}'")
    if filters:
        print(f"   Filtered by: {filters}")
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"{i+1}. {metadata['title']} ({metadata['language']})")
        print(f"   Why: {doc[:100]}...")
        print()

# Test your recommendation system
recommend_movies("I love magical adventures with young heroes")
recommend_movies("I want action movies with fighting")
recommend_movies("technology and programming", filters={"category": "tech"})

# Ejercicio adicional: Sistema básico de BD vectorial
class BaseDatosVectorial:
    def __init__(self):
        self.vectores = []
        self.metadatos = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def agregar(self, texto, metadata=None):
        vector = self.model.encode([texto])[0]
        self.vectores.append(vector)
        self.metadatos.append(metadata or {"texto": texto})
    
    def buscar(self, consulta, k=3):
        query_vector = self.model.encode([consulta])[0]
        similarities = []
        
        for i, vector in enumerate(self.vectores):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((similarity, i))
        
        # Ordenar por similitud y retornar top k
        similarities.sort(reverse=True)
        return [(self.metadatos[idx], sim) for sim, idx in similarities[:k]]

# Demo de la clase personalizada
bd = BaseDatosVectorial()
categorias = ['tecnología', 'deportes', 'comida', 'viajes', 'ciencia']
frases_ejemplo = [
    "La inteligencia artificial está transformando el mundo",
    "El fútbol es el deporte más popular del mundo",
    "La pizza italiana es deliciosa",
    "Viajar por Europa es una experiencia increíble",
    "La física cuántica es fascinante"
]

for frase, cat in zip(frases_ejemplo, categorias):
    bd.agregar(frase, {"categoria": cat, "texto": frase})

print("\n🔍 Búsqueda en BD personalizada:")
resultados = bd.buscar("deportes y competencias")
for metadata, score in resultados:
    print(f"  {metadata['categoria']}: {metadata['texto'][:50]}... (Score: {score:.3f})")
```

### 🏆 Quest Completion Requirements
1. **Bronze**: Get the system working and test 3 different preferences
2. **Silver**: Add metadata filtering (genre, year, rating) and use it in queries  
3. **Gold**: Create 50+ movies with detailed metadata and implement "similar users" feature
4. **Legendary**: Add update/delete functionality and implement user rating system

### Entregables
- Clase `BaseDatosVectorial` funcional
- Dataset con 50 frases categorizadas
- Demo que muestre búsquedas semánticas funcionando

---

## 🎮 **Quest 3: Optimized Vector Search** - *The Acceleration*
**Power Level**: Intermediate  
**Time**: 4-6 hours  
**New Concepts**: FAISS, Indexing, Performance Optimization, Batch Operations

### 🎪 The Challenge: "Lightning-Fast News Search Engine"
Handle 10,000+ news articles with sub-millisecond search times!

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json

class LightningNewsSearch:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.dimension = 384  # Model output dimension
        
        # Create FAISS index for lightning-fast search
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = Cosine for normalized vectors
        self.articles = []
        self.metadata = []
    
    def add_articles_batch(self, articles, metadata_list):
        """Add thousands of articles in one lightning strike"""
        print(f"⚡ Encoding {len(articles)} articles...")
        
        # Generate embeddings in batch (much faster!)
        embeddings = self.model.encode(articles, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store articles and metadata
        self.articles.extend(articles)
        self.metadata.extend(metadata_list)
        
        print(f"⚡ Database now contains {len(self.articles)} articles")
    
    def lightning_search(self, query, k=5, category_filter=None):
        """Search at the speed of light!"""
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index - this is where the magic happens!
        distances, indices = self.index.search(query_embedding.astype('float32'), k * 2)  # Get extra in case we filter
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if len(results) >= k:
                break
                
            # Apply category filter if specified
            if category_filter and self.metadata[idx].get('category') != category_filter:
                continue
                
            results.append({
                'article': self.articles[idx],
                'metadata': self.metadata[idx],
                'similarity': float(distance),
                'rank': len(results) + 1
            })
        
        search_time = time.time() - start_time
        return results, search_time

# Clase optimizada con FAISS (del roadmap original)
class BaseDatosVectorialOptimizada:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        self.metadatos = []
    
    def agregar_lote(self, vectores, metadatos):
        """Agregar múltiples vectores de una vez"""
        vectores_np = np.array(vectores).astype('float32')
        faiss.normalize_L2(vectores_np)
        self.index.add(vectores_np)
        self.metadatos.extend(metadatos)
    
    def buscar_con_filtros(self, consulta, filtros=None, k=5):
        """Búsqueda con filtros por metadata"""
        # Normalizar consulta
        consulta_np = np.array([consulta]).astype('float32')
        faiss.normalize_L2(consulta_np)
        
        # Buscar en índice
        distances, indices = self.index.search(consulta_np, k * 2)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if len(results) >= k:
                break
            
            metadata = self.metadatos[idx]
            
            # Aplicar filtros si se especifican
            if filtros:
                if all(metadata.get(key) == value for key, value in filtros.items()):
                    results.append({
                        'metadata': metadata,
                        'similarity': float(dist),
                        'index': int(idx)
                    })
            else:
                results.append({
                    'metadata': metadata,
                    'similarity': float(dist),
                    'index': int(idx)
                })
        
        return results

# Create sample news database
def generate_sample_news():
    """Generate realistic news articles for testing"""
    tech_articles = [
        "New AI breakthrough enables computers to understand human emotions better",
        "Quantum computing startup raises $100M to revolutionize encryption",
        "Electric vehicle sales surge as battery technology improves",
        "Social media platform introduces new privacy features for users"
    ]
    
    sports_articles = [
        "Championship game ends in dramatic overtime victory",
        "Young athlete breaks 20-year-old world record in swimming",
        "Soccer world cup preparations underway in host country",
        "Basketball legend announces retirement after stellar career"
    ]
    
    # Generate more articles programmatically
    articles = tech_articles + sports_articles
    metadata = [
        {'category': 'tech', 'date': '2024-01-01', 'source': 'TechNews'},
        {'category': 'tech', 'date': '2024-01-02', 'source': 'Innovation Daily'},
        {'category': 'tech', 'date': '2024-01-03', 'source': 'FutureTech'},
        {'category': 'tech', 'date': '2024-01-04', 'source': 'DigitalWorld'},
        {'category': 'sports', 'date': '2024-01-01', 'source': 'SportsCenter'},
        {'category': 'sports', 'date': '2024-01-02', 'source': 'AthleteNews'},
        {'category': 'sports', 'date': '2024-01-03', 'source': 'Championship Weekly'},
        {'category': 'sports', 'date': '2024-01-04', 'source': 'Sports Tribune'}
    ]
    
    return articles, metadata

# Initialize and test your lightning search engine
news_engine = LightningNewsSearch()
articles, metadata = generate_sample_news()
news_engine.add_articles_batch(articles, metadata)

# Test lightning-fast searches with performance metrics
def test_search_performance():
    queries = [
        "artificial intelligence and machine learning",
        "championship sports competition", 
        "technology innovation breakthrough"
    ]
    
    print("\n📊 Performance Benchmarks:")
    total_time = 0
    
    for query in queries:
        results, search_time = news_engine.lightning_search(query, k=3)
        total_time += search_time
        
        print(f"🔍 Query: '{query}'")
        print(f"⚡ Search time: {search_time*1000:.2f}ms")
        
        for result in results:
            print(f"  {result['rank']}. {result['metadata']['category'].upper()}: {result['article'][:80]}...")
            print(f"     Similarity: {result['similarity']:.3f}")
        print()
    
    print(f"📈 Average search time: {(total_time/len(queries))*1000:.2f}ms")

test_search_performance()
```

### 🏆 Quest Completion Requirements
1. **Bronze**: Get FAISS working and compare speed with naive search on 1000 articles
2. **Silver**: Implement category filtering and batch updates
3. **Gold**: Add multiple index types (HNSW, IVF) and benchmark performance
4. **Legendary**: Create a REST API with async endpoints and implement index persistence

### Entregables
- BD vectorial optimizada con FAISS
- Sistema de filtros funcional
- Benchmarks de rendimiento comparando con implementación naive
- Dataset expandido a 500 documentos

---

## 🎮 **Quest 4: RAG System Master** - *The Intelligence*
**Power Level**: Advanced  
**Time**: 6-8 hours  
**New Concepts**: RAG Pipeline, LLM Integration, Context Management, Evaluation

### 🎪 The Challenge: "AI Knowledge Oracle"
Build a system that can answer complex questions by finding and synthesizing information from documents!

```python
import chromadb
from chromadb.utils import embedding_functions
import openai  # or use ollama for local models
import re
from typing import List, Dict, Tuple

class KnowledgeOracle:
    def __init__(self, openai_api_key=None):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./oracle_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create knowledge base collection
        self.knowledge_base = self.client.get_or_create_collection(
            name="oracle_knowledge",
            embedding_function=self.embedding_function
        )
        
        # Initialize OpenAI (or replace with local model)
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def ingest_document(self, text: str, doc_id: str, metadata: Dict = None):
        """Break document into chunks and add to knowledge base"""
        chunks = self._chunk_document(text, self.chunk_size, self.chunk_overlap)
        
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadata = [
            {**(metadata or {}), "chunk_id": i, "doc_id": doc_id} 
            for i in range(len(chunks))
        ]
        
        self.knowledge_base.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=chunk_metadata
        )
        
        print(f"📚 Ingested document '{doc_id}' as {len(chunks)} chunks")
    
    def _chunk_document(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Smart chunking that preserves sentence boundaries"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def ask_oracle(self, question: str, num_contexts: int = 3) -> Dict:
        """The main RAG pipeline - retrieve and generate"""
        # Step 1: Retrieve relevant contexts
        contexts = self._retrieve_contexts(question, num_contexts)
        
        if not contexts:
            return {
                "answer": "I don't have enough information to answer that question.",
                "contexts": [],
                "confidence": 0.0
            }
        
        # Step 2: Generate answer using LLM
        answer = self._generate_answer(question, contexts)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "confidence": self._calculate_confidence(contexts)
        }
    
    def _retrieve_contexts(self, question: str, k: int) -> List[Dict]:
        """Retrieve the most relevant contexts for the question"""
        results = self.knowledge_base.query(
            query_texts=[question],
            n_results=k
        )
        
        contexts = []
        for doc, metadata, distance in zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ):
            contexts.append({
                "text": doc,
                "metadata": metadata,
                "relevance_score": 1 - distance  # Convert distance to similarity
            })
        
        return contexts
    
    def _generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate answer using retrieved contexts"""
        # Prepare context for the LLM
        context_text = "\n\n".join([
            f"Context {i+1}: {ctx['text']}" 
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""Based on the following contexts, please answer the question. If the contexts don't contain enough information to answer the question, say so clearly.

Question: {question}

Contexts:
{context_text}

Answer:"""
        
        try:
            # Using OpenAI GPT (replace with your preferred model)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Fallback to simple context concatenation if LLM fails
            return f"Based on the available information: {contexts[0]['text'][:200]}..."
    
    def _calculate_confidence(self, contexts: List[Dict]) -> float:
        """Calculate confidence based on context relevance"""
        if not contexts:
            return 0.0
        
        avg_relevance = sum(ctx['relevance_score'] for ctx in contexts) / len(contexts)
        return min(avg_relevance, 1.0)

# Sistema RAG completo (del roadmap original)
class SistemaRAG:
    def __init__(self, bd_vectorial, modelo_llm=None):
        self.bd = bd_vectorial
        self.llm = modelo_llm
    
    def dividir_documento(self, texto, tamaño_chunk=500, overlap=50):
        """Dividir manteniendo contexto"""
        sentences = re.split(r'[.!?]+', texto)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > tamaño_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def responder_pregunta(self, pregunta):
        """Pipeline RAG completo"""
        # 1. Buscar contexto relevante
        contextos = self.bd.buscar(pregunta, k=3)
        
        # 2. Construir prompt
        prompt = self.construir_prompt(pregunta, contextos)
        
        # 3. Generar respuesta
        if self.llm:
            respuesta = self.llm.generar(prompt)
        else:
            # Fallback simple
            respuesta = f"Basado en el contexto: {contextos[0][0]['texto'] if contextos else 'No hay contexto'}"
        
        return respuesta, contextos
    
    def construir_prompt(self, pregunta, contextos):
        context_text = "\n".join([ctx[0]['texto'] for ctx in contextos])
        return f"Pregunta: {pregunta}\nContexto: {context_text}\nRespuesta:"

# Sample knowledge base - load it with interesting content!
sample_documents = {
    "vector_databases": """
    Vector databases are specialized databases designed to store and query high-dimensional vectors.
    Unlike traditional databases that store structured data in rows and columns, vector databases
    store embeddings - numerical representations of unstructured data like text, images, or audio.
    
    The key advantage of vector databases is their ability to perform similarity searches.
    Instead of exact matches, they find items that are semantically similar based on the
    mathematical distance between their vector representations.
    
    Popular vector databases include Pinecone, Weaviate, Chroma, and Qdrant. Each offers
    different features like distributed computing, real-time updates, and various indexing algorithms.
    """,
    
    "machine_learning": """
    Machine learning is a subset of artificial intelligence that enables computers to learn
    and make decisions from data without being explicitly programmed for every scenario.
    
    There are three main types of machine learning:
    1. Supervised learning: Learning from labeled examples
    2. Unsupervised learning: Finding patterns in unlabeled data  
    3. Reinforcement learning: Learning through interaction and feedback
    
    Common applications include recommendation systems, image recognition, natural language
    processing, and autonomous vehicles. The field has grown rapidly with the availability
    of big data and powerful computing resources.
    """,
    
    "fotosintesis": """
    La fotosíntesis es el proceso mediante el cual las plantas verdes, las algas y algunas bacterias 
    utilizan la energía de la luz solar para convertir el dióxido de carbono y el agua en glucosa y oxígeno.
    """,
    
    "jupiter": """
    Júpiter es el planeta más grande de nuestro sistema solar. Es un gigante gaseoso compuesto 
    principalmente de hidrógeno y helio, y es conocido por su Gran Mancha Roja.
    """
}

# Initialize and test your Knowledge Oracle
def demo_knowledge_oracle():
    oracle = KnowledgeOracle()  # Add your OpenAI key here if you have one
    
    # Ingest sample documents
    for doc_id, content in sample_documents.items():
        oracle.ingest_document(content, doc_id, {"topic": doc_id})
    
    # Test questions
    test_questions = [
        "What are vector databases and how do they work?",
        "What are the main types of machine learning?",
        "How do vector databases differ from traditional databases?",
        "What are some applications of machine learning?",
        "¿Qué es la fotosíntesis?",
        "¿Cuál es el planeta más grande?"
    ]
    
    print("🔮 Knowledge Oracle Demo")
    print("=" * 50)
    
    for question in test_questions:
        result = oracle.ask_oracle(question)
        
        print(f"\n❓ Question: {question}")
        print(f"🔮 Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"📚 Sources used: {len(result['contexts'])}")
        print("-" * 50)

demo_knowledge_oracle()

# Sistema Q&A básico (del roadmap original)
def sistema_qa_basico():
    client = chromadb.Client()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    qa_collection = client.get_or_create_collection(
        name="base_de_conocimiento_wiki",
        embedding_function=sentence_transformer_ef
    )

    # Fragmentos de conocimiento
    contexto = [
        sample_documents["fotosintesis"],
        sample_documents["jupiter"],
        "La inteligencia artificial (IA) es un campo de la informática dedicado a la creación de máquinas que pueden realizar tareas que normalmente requieren inteligencia humana.",
        "La capital de Francia es París, famosa por monumentos como la Torre Eiffel, el Museo del Louvre y la Catedral de Notre Dame.",
        "El agua (H2O) es una molécula polar y es considerada el 'solvente universal'."
    ]

    # Añadir el conocimiento a nuestra base de datos vectorial
    if qa_collection.count() == 0:
        qa_collection.add(
            documents=contexto,
            ids=[f"info{i}" for i in range(len(contexto))]
        )
        print("Base de conocimiento cargada.")

    def responder_pregunta(pregunta):
        results = qa_collection.query(
            query_texts=[pregunta],
            n_results=1
        )
        contexto_relevante = results['documents'][0][0]
        print(f"\nPregunta: {pregunta}")
        print(f"Respuesta Encontrada: \n{contexto_relevante}")

    # Pruebas
    responder_pregunta("¿Qué es la IA?")
    responder_pregunta("¿Cuál es el planeta más grande?")
    responder_pregunta("¿Qué proceso usan las plantas para obtener energía?")
    responder_pregunta("¿Dónde está la Torre Eiffel?")

print("\n" + "="*60)
print("SISTEMA Q&A BÁSICO")
print("="*60)
sistema_qa_basico()
```

### 🏆 Quest Completion Requirements
1. **Bronze**: Get basic RAG working with sample documents
2. **Silver**: Add document upload, multiple file formats (PDF, TXT), and metadata filtering
3. **Gold**: Implement re-ranking, source attribution, and evaluation metrics
4. **Legendary**: Add multi-modal support (images, audio) and create a web interface

### Entregables
- Sistema RAG funcional
- Corpus de al menos 100 documentos procesados
- 50 preguntas de evaluación con respuestas esperadas
- Interfaz simple (CLI o web básica) para hacer preguntas

---

## 🎮 **Quest 5: Multimodal Vector Master** - *The Transcendence*
**Power Level**: Expert  
**Time**: 10-15 hours  
**New Concepts**: Multimodal Embeddings, CLIP, Cross-Modal Search, Production Systems

### 🎪 The Challenge: "Omni-Search Engine"
Build a system that can search across text, images, and audio using natural language queries!

```python
import chromadb
from chromadb.utils import embedding_functions
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union, List, Dict
import os

class OmniSearchEngine:
    def __init__(self):
        # Initialize ChromaDB with separate collections for each modality
        self.client = chromadb.PersistentClient(path="./omni_search_db")
        
        # Load CLIP for multimodal embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Custom embedding function for CLIP
        self.clip_embedding_function = self._create_clip_embedding_function()
        
        # Create collections for different data types
        self.text_collection = self.client.get_or_create_collection(
            name="text_content",
            embedding_function=self.clip_embedding_function
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name="image_content", 
            embedding_function=self.clip_embedding_function
        )
        
        print("🌟 OmniSearch Engine initialized!")
    
    def _create_clip_embedding_function(self):
        """Create custom embedding function using CLIP"""
        class CLIPEmbeddingFunction:
            def __init__(self, model, processor):
                self.model = model
                self.processor = processor
            
            def __call__(self, texts):
                # CLIP can handle both text and image paths
                embeddings = []
                for text in texts:
                    if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # It's an image path
                        image = Image.open(text)
                        inputs = self.processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            image_features = self.model.get_image_features(**inputs)
                            embedding = image_features.numpy().flatten()
                    else:
                        # It's text
                        inputs = self.processor(text=text, return_tensors="pt", truncation=True)
                        with torch.no_grad():
                            text_features = self.model.get_text_features(**inputs)
                            embedding = text_features.numpy().flatten()
                    
                    embeddings.append(embedding.tolist())
                
                return embeddings
        
        return CLIPEmbeddingFunction(self.clip_model, self.clip_processor)
    
    def add_text_content(self, texts: List[str], metadata: List[Dict] = None):
        """Add text content to the search engine"""
        ids = [f"text_{i}_{hash(text) % 10000}" for i, text in enumerate(texts)]
        
        self.text_collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata or [{"type": "text"} for _ in texts]
        )
        
        print(f"📝 Added {len(texts)} text documents")
    
    def add_image_content(self, image_paths: List[str], descriptions: List[str] = None, metadata: List[Dict] = None):
        """Add images to the search engine"""
        if descriptions is None:
            descriptions = [f"Image: {os.path.basename(path)}" for path in image_paths]
        
        ids = [f"image_{i}_{hash(path) % 10000}" for i, path in enumerate(image_paths)]
        
        # Store image paths as documents for CLIP processing
        self.image_collection.add(
            documents=image_paths,  # CLIP embedding function will detect these are image paths
            ids=ids,
            metadatas=metadata or [{"type": "image", "description": desc} for desc in descriptions]
        )
        
        print(f"🖼️ Added {len(image_paths)} images")
    
    def omni_search(self, query: str, search_types: List[str] = ["text", "image"], k: int = 5) -> Dict:
        """Search across all modalities with a single query"""
        results = {"query": query, "results": []}
        
        if "text" in search_types:
            text_results = self.text_collection.query(
                query_texts=[query],
                n_results=k
            )
            
            for doc, metadata, distance in zip(
                text_results['documents'][0],
                text_results['metadatas'][0], 
                text_results['distances'][0]
            ):
                results["results"].append({
                    "type": "text",
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance,
                    "score": (1 - distance) * 100
                })
        
        if "image" in search_types:
            image_results = self.image_collection.query(
                query_texts=[query],
                n_results=k
            )
            
            for doc, metadata, distance in zip(
                image_results['documents'][0],
                image_results['metadatas'][0],
                image_results['distances'][0]
            ):
                results["results"].append({
                    "type": "image", 
                    "content": doc,  # This is the image path
                    "metadata": metadata,
                    "similarity": 1 - distance,
                    "score": (1 - distance) * 100
                })
        
        # Sort all results by similarity
        results["results"].sort(key=lambda x: x["similarity"], reverse=True)
        results["results"] = results["results"][:k]
        
        return results
    
    def cross_modal_search(self, image_path: str, k: int = 5) -> Dict:
        """Search for text content using an image as query"""
        # Use the image to search text collection
        text_results = self.text_collection.query(
            query_texts=[image_path],  # CLIP will process this as an image
            n_results=k
        )
        
        results = {"image_query": image_path, "text_matches": []}
        
        for doc, metadata, distance in zip(
            text_results['documents'][0],
            text_results['metadatas'][0],
            text_results['distances'][0]
        ):
            results["text_matches"].append({
                "text": doc,
                "metadata": metadata,
                "similarity": 1 - distance
            })
        
        return results

# Base de datos vectorial multimodal (del roadmap original)
class BDVectorialMultimodal:
    def __init__(self):
        self.indices = {
            'texto': faiss.IndexFlatIP(384),
            'imagen': faiss.IndexFlatIP(512),
            'audio': faiss.IndexFlatIP(256)
        }
        self.metadatos = {'texto': [], 'imagen': [], 'audio': []}
    
    def agregar_multimedia(self, contenido, tipo, metadata):
        """Generar embeddings según el tipo"""
        # Placeholder - implementar según el tipo de contenido
        if tipo == 'texto':
            # Usar sentence transformers
            pass
        elif tipo == 'imagen':
            # Usar CLIP o similar
            pass
        elif tipo == 'audio':
            # Usar Whisper embeddings o similar
            pass
    
    def buscar_multimodal(self, query, tipos=['texto'], k=5):
        """Búsqueda cross-modal"""
        results = []
        for tipo in tipos:
            # Implementar búsqueda en cada modalidad
            pass
        return results
    
    def rerank_resultados(self, query, resultados_iniciales):
        """Usar modelo específico para re-ordenar"""
        # Implementar re-ranking
        return resultados_iniciales
    
    def cache_consultas_frecuentes(self):
        """Optimizar consultas repetidas"""
        # Implementar caché inteligente
        pass
    
    def actualizar_vectores(self, contenido_modificado):
        """Actualizar sin reconstruir todo el índice"""
        # Implementar actualización incremental
        pass

# Demo the Omni-Search Engine
def demo_omni_search():
    engine = OmniSearchEngine()
    
    # Sample text content
    sample_texts = [
        "A beautiful sunset over the ocean with orange and pink colors",
        "A cute golden retriever playing in a park with children",
        "Modern architecture with glass buildings and urban design",
        "Delicious Italian pasta with tomato sauce and fresh basil",
        "Advanced artificial intelligence and machine learning algorithms",
        "Space exploration and rockets launching to Mars"
    ]
    
    text_metadata = [
        {"category": "nature", "topic": "sunset"},
        {"category": "animals", "topic": "dogs"},
        {"category": "architecture", "topic": "buildings"},
        {"category": "food", "topic": "italian"},
        {"category": "technology", "topic": "AI"},
        {"category": "science", "topic": "space"}
    ]
    
    # Add content to the engine
    engine.add_text_content(sample_texts, text_metadata)
    
    # Test searches
    test_queries = [
        "beautiful landscapes and nature scenes",
        "cute animals and pets", 
        "modern technology and innovation",
        "delicious food and cooking"
    ]
    
    print("\n🔍 OmniSearch Demo Results")
    print("=" * 60)
    
    for query in test_queries:
        results = engine.omni_search(query, k=3)
        
        print(f"\n🎯 Query: '{query}'")
        print("-" * 40)
        
        for i, result in enumerate(results["results"], 1):
            print(f"{i}. [{result['type'].upper()}] Score: {result['score']:.1f}")
            print(f"   Content: {result['content'][:80]}...")
            print(f"   Category: {result['metadata'].get('category', 'N/A')}")
            print()

demo_omni_search()
```

### 🏆 Quest Completion Requirements
1. **Bronze**: Get CLIP working with text-image cross-modal search
2. **Silver**: Add audio support using Whisper embeddings, implement semantic clustering
3. **Gold**: Build production API with async processing, caching, and monitoring
4. **Legendary**: Deploy to cloud with auto-scaling, implement A/B testing for different models

### Entregables
- Sistema multimodal completo
- Dataset con 1000+ elementos (texto, imágenes, audio)
- Suite de evaluación automática
- Documentación técnica completa
- API REST para el sistema
- Dashboard de métricas y monitoreo

---

## 🎖️ **Final Boss Challenge: Production Vector Database**

Once you've completed all quests, take on the ultimate challenge:

### The Mission: "Vector Database as a Service"
Build a complete production-ready vector database service with:

- **Multi-tenant architecture** with isolated collections
- **REST API** with authentication and rate limiting  
- **Real-time updates** with streaming ingestion
- **Monitoring and alerting** with performance metrics
- **Auto-scaling** based on load
- **Backup and disaster recovery**
- **A/B testing** for different embedding models
- **Cost optimization** with intelligent caching

---

## 🛠️ **Herramientas y Tecnologías Recomendadas**

### Librerías Python Esenciales
- **sentence-transformers**: Embeddings de texto
- **faiss-cpu/faiss-gpu**: Búsqueda vectorial eficiente
- **numpy**: Operaciones matemáticas
- **pandas**: Manipulación de datos
- **scikit-learn**: Métricas y utilidades ML
- **chromadb**: Base de datos vectorial simple

### Bases de Datos Vectoriales Comerciales
- **Pinecone**: Servicio cloud managed
- **Weaviate**: Open source con GraphQL
- **Chroma**: Simple y local-first
- **Qdrant**: High-performance con Rust
- **Milvus**: Enterprise-grade, open source

### Modelos de Embeddings
- **Texto**: sentence-transformers, OpenAI embeddings, Cohere
- **Imágenes**: CLIP, BLIP-2
- **Audio**: Wav2Vec2, Whisper embeddings
- **Multimodal**: CLIP, BLIP, LLaVA

## 🎯 Proyectos Adicionales (Opcional)

1. **ChatBot especializado**: RAG para dominio específico (medicina, derecho, finanzas)
2. **Motor de búsqueda de código**: Búsqueda semántica en repositorios
3. **Sistema de recomendación de contenido**: Para plataforma de streaming/e-commerce
4. **Asistente de investigación**: Para papers académicos y literatura científica

## 🎯 Skills Unlocked by Quest Completion

| Quest | Technical Skills | Business Value |
|-------|-----------------|----------------|
| 1 | Vector math, similarity metrics | Understanding of recommendation systems |
| 2 | ChromaDB, embeddings, collections | Content management and search |
| 3 | FAISS, optimization, indexing | High-performance applications |
| 4 | RAG, LLM integration, evaluation | AI-powered Q&A systems |
| 5 | Multimodal AI, CLIP, production systems | Next-gen search experiences |

## 📈 Habilidades que Desarrollarás

- Comprensión profunda de embeddings y espacios vectoriales
- Optimización de sistemas de búsqueda a gran escala
- Evaluación y mejora de sistemas de IA
- Arquitectura de sistemas de información modernos
- Integración de múltiples modelos de ML/AI

---

## 🚀 Your Hero's Journey Awaits!

Each quest builds upon the previous one, ensuring you develop both theoretical understanding and practical skills. The challenges are designed to be fun while maintaining technical rigor.

**Ready to begin your transformation from zero to vector database hero?** 

Start with Quest 1 and unlock your potential! 🌟

---

*May the vectors be with you!* ⚡️

¡Éxito en tu aprendizaje! 🚀