#!/usr/bin/env python3
"""
Pre-Quest Example 3: Different Types of Embeddings
Comparación de diferentes tipos de embeddings y sus casos de uso
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def ejemplo_bag_of_words():
    """
    🎯 Objetivo: Entender Bag of Words (BoW)
    """
    print("📊 EJEMPLO 1: Bag of Words (BoW)")
    print("=" * 60)
    
    # Documentos de ejemplo
    documentos = [
        "el gato come pescado",
        "el perro come carne",
        "gato y perro son animales",
        "pescado y carne son comida"
    ]
    
    print("📝 Documentos:")
    for i, doc in enumerate(documentos):
        print(f"   {i+1}. '{doc}'")
    
    # Crear vectorizer BoW
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(documentos)
    
    # Mostrar vocabulario
    vocabulario = vectorizer.get_feature_names_out()
    print(f"\n📚 Vocabulario: {list(vocabulario)}")
    
    # Mostrar matriz BoW
    print("\n🔢 Matriz Bag of Words:")
    print("   Filas = documentos, Columnas = palabras")
    print("   Valores = frecuencia de cada palabra")
    
    bow_array = bow_matrix.toarray()
    
    # Encabezados
    print("   " + " ".join([f"{word:8s}" for word in vocabulario]))
    
    for i, fila in enumerate(bow_array):
        print(f"{i+1}: " + " ".join([f"{val:8d}" for val in fila]))
    
    # Calcular similitudes
    print("\n🔍 Similitudes entre documentos:")
    similitudes = cosine_similarity(bow_array)
    
    for i in range(len(documentos)):
        for j in range(i+1, len(documentos)):
            print(f"   Doc {i+1} ↔ Doc {j+1}: {similitudes[i, j]:.3f}")
    
    return bow_array, vocabulario

def ejemplo_tfidf():
    """
    🎯 Objetivo: Entender TF-IDF (Term Frequency - Inverse Document Frequency)
    """
    print("\n\n📈 EJEMPLO 2: TF-IDF")
    print("=" * 60)
    
    # Documentos donde algunas palabras son más comunes
    documentos = [
        "el gato come pescado fresco",
        "el perro come carne roja",
        "el gato y el perro son animales domésticos",
        "pescado fresco y carne roja son comida nutritiva",
        "el agua es importante para todos los animales"
    ]
    
    print("📝 Documentos:")
    for i, doc in enumerate(documentos):
        print(f"   {i+1}. '{doc}'")
    
    # Crear vectorizer TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos)
    
    vocabulario = tfidf_vectorizer.get_feature_names_out()
    print(f"\n📚 Vocabulario: {list(vocabulario)}")
    
    # Mostrar matriz TF-IDF (solo valores significativos)
    print("\n🔢 Matriz TF-IDF (valores > 0.1):")
    tfidf_array = tfidf_matrix.toarray()
    
    for i, fila in enumerate(tfidf_array):
        print(f"\nDoc {i+1}:")
        for j, val in enumerate(fila):
            if val > 0.1:
                print(f"   {vocabulario[j]}: {val:.3f}")
    
    # Explicar TF-IDF
    print("\n💡 ¿Qué significa TF-IDF?")
    print("   • TF (Term Frequency): Frecuencia del término en el documento")
    print("   • IDF (Inverse Document Frequency): Rareza del término en la colección")
    print("   • TF-IDF = TF × IDF")
    print("   • Palabras comunes (como 'el') tienen IDF bajo")
    print("   • Palabras raras (como 'nutritiva') tienen IDF alto")
    
    return tfidf_array, vocabulario

def ejemplo_word2vec_simulado():
    """
    🎯 Objetivo: Simular embeddings densos como Word2Vec
    """
    print("\n\n🧠 EJEMPLO 3: Embeddings Densos (simulando Word2Vec)")
    print("=" * 60)
    
    # Simular embeddings densos de palabras
    # En la realidad, estos vienen de entrenar redes neuronales
    embeddings_palabras = {
        "gato": np.array([0.8, 0.1, 0.9, 0.2, 0.7]),
        "felino": np.array([0.7, 0.0, 0.8, 0.1, 0.8]),  # Similar a gato
        "perro": np.array([0.9, 0.0, 0.7, 0.3, 0.6]),
        "canino": np.array([0.8, 0.1, 0.6, 0.2, 0.7]),  # Similar a perro
        "pescado": np.array([0.1, 0.9, 0.2, 0.8, 0.1]),
        "carne": np.array([0.2, 0.8, 0.1, 0.9, 0.2]),
        "mesa": np.array([0.0, 0.0, 0.1, 0.0, 0.0]),     # Muy diferente
    }
    
    print("🔢 Embeddings de palabras (vectores densos):")
    for palabra, embedding in embeddings_palabras.items():
        print(f"   {palabra:8s}: {embedding}")
    
    # Calcular similitudes
    print("\n🔍 Similitudes entre palabras:")
    palabras = list(embeddings_palabras.keys())
    
    for i, palabra1 in enumerate(palabras):
        for j, palabra2 in enumerate(palabras[i+1:], i+1):
            emb1 = embeddings_palabras[palabra1]
            emb2 = embeddings_palabras[palabra2]
            
            similitud = cosine_similarity([emb1], [emb2])[0][0]
            print(f"   {palabra1:8s} ↔ {palabra2:8s}: {similitud:.3f}")
    
    print("\n💡 Observaciones:")
    print("   • 'gato' y 'felino' son muy similares (0.9+)")
    print("   • 'perro' y 'canino' son muy similares (0.9+)")
    print("   • 'pescado' y 'carne' son medianamente similares (comida)")
    print("   • 'mesa' es diferente a todo (objeto vs animales/comida)")

def ejemplo_embeddings_contextuales():
    """
    🎯 Objetivo: Simular embeddings contextuales
    """
    print("\n\n🎭 EJEMPLO 4: Embeddings Contextuales")
    print("=" * 60)
    
    # La misma palabra puede tener diferentes vectores según el contexto
    contextos = {
        "banco_financiero": {
            "frase": "Voy al banco a depositar dinero",
            "vector": np.array([0.8, 0.2, 0.9, 0.1, 0.7])
        },
        "banco_asiento": {
            "frase": "Me siento en el banco del parque",
            "vector": np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        },
        "banco_peces": {
            "frase": "Vimos un banco de peces en el mar",
            "vector": np.array([0.3, 0.1, 0.2, 0.9, 0.8])
        }
    }
    
    print("🎭 La palabra 'banco' en diferentes contextos:")
    
    for contexto, info in contextos.items():
        print(f"\n📝 {contexto}:")
        print(f"   Frase: '{info['frase']}'")
        print(f"   Vector: {info['vector']}")
    
    # Calcular similitudes entre contextos
    print("\n🔍 Similitudes entre diferentes usos de 'banco':")
    
    contextos_list = list(contextos.keys())
    for i, ctx1 in enumerate(contextos_list):
        for j, ctx2 in enumerate(contextos_list[i+1:], i+1):
            vec1 = contextos[ctx1]['vector']
            vec2 = contextos[ctx2]['vector']
            
            similitud = cosine_similarity([vec1], [vec2])[0][0]
            print(f"   {ctx1} ↔ {ctx2}: {similitud:.3f}")
    
    print("\n💡 Ventajas de embeddings contextuales:")
    print("   • La misma palabra puede tener vectores diferentes")
    print("   • El significado depende del contexto")
    print("   • Capturan polisemia (múltiples significados)")
    print("   • Modelos como BERT, GPT usan este enfoque")

def comparar_todos_los_metodos():
    """
    🎯 Objetivo: Comparar todos los métodos de vectorización
    """
    print("\n\n⚖️ EJEMPLO 5: Comparación de Métodos")
    print("=" * 60)
    
    # Frase de prueba
    frase_consulta = "animal doméstico"
    documentos = [
        "gato animal doméstico",
        "felino mascota casa",
        "perro animal leal", 
        "mesa mueble madera",
        "coche vehículo rápido"
    ]
    
    print(f"🔎 Consulta: '{frase_consulta}'")
    print("📚 Documentos:")
    for i, doc in enumerate(documentos):
        print(f"   {i+1}. '{doc}'")
    
    # Método 1: Bag of Words
    print("\n📊 1. Bag of Words:")
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(documentos + [frase_consulta])
    bow_similitudes = cosine_similarity([bow_matrix[-1]], bow_matrix[:-1])[0]
    
    for i, sim in enumerate(bow_similitudes):
        print(f"   Doc {i+1}: {sim:.3f}")
    
    # Método 2: TF-IDF
    print("\n📈 2. TF-IDF:")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos + [frase_consulta])
    tfidf_similitudes = cosine_similarity([tfidf_matrix[-1]], tfidf_matrix[:-1])[0]
    
    for i, sim in enumerate(tfidf_similitudes):
        print(f"   Doc {i+1}: {sim:.3f}")
    
    # Método 3: Embeddings densos (simulados)
    print("\n🧠 3. Embeddings Densos (simulados):")
    
    # Simular embeddings más inteligentes
    embeddings_docs = np.array([
        [0.9, 0.8, 0.1, 0.2, 0.9],  # gato animal doméstico
        [0.8, 0.9, 0.0, 0.1, 0.8],  # felino mascota casa
        [0.7, 0.7, 0.2, 0.3, 0.8],  # perro animal leal
        [0.0, 0.1, 0.9, 0.8, 0.0],  # mesa mueble madera
        [0.1, 0.0, 0.8, 0.9, 0.1],  # coche vehículo rápido
    ])
    
    embedding_consulta = np.array([0.8, 0.8, 0.1, 0.2, 0.9])  # animal doméstico
    
    dense_similitudes = cosine_similarity([embedding_consulta], embeddings_docs)[0]
    
    for i, sim in enumerate(dense_similitudes):
        print(f"   Doc {i+1}: {sim:.3f}")
    
    # Resumen
    print("\n🏆 Mejor match por método:")
    print(f"   BoW: Doc {np.argmax(bow_similitudes) + 1} ({np.max(bow_similitudes):.3f})")
    print(f"   TF-IDF: Doc {np.argmax(tfidf_similitudes) + 1} ({np.max(tfidf_similitudes):.3f})")
    print(f"   Dense: Doc {np.argmax(dense_similitudes) + 1} ({np.max(dense_similitudes):.3f})")

def main():
    """
    Ejecutar todos los ejemplos
    """
    print("🔬 PRE-QUEST: Tipos de Embeddings")
    print("🎯 Comparación de diferentes métodos de vectorización")
    print("=" * 80)
    
    # Ejecutar ejemplos
    ejemplo_bag_of_words()
    ejemplo_tfidf()
    ejemplo_word2vec_simulado()
    ejemplo_embeddings_contextuales()
    comparar_todos_los_metodos()
    
    print("\n\n🎉 ¡Comparación de embeddings completada!")
    print("💡 Conceptos clave aprendidos:")
    print("   • Bag of Words: Conteo simple de palabras")
    print("   • TF-IDF: Pondera importancia de términos")
    print("   • Word2Vec: Embeddings densos entrenados")
    print("   • Embeddings contextuales: Vectores dependientes del contexto")
    print("   • Trade-offs entre interpretabilidad y capacidad semántica")
    
    print("\n📊 Resumen de métodos:")
    print("   • Sparse (BoW, TF-IDF): Interpretables, dimensión alta")
    print("   • Dense (Word2Vec, BERT): Semánticos, dimensión baja")
    print("   • Contextuales: Capturan polisemia y contexto")
    
    print("\n🎮 ¡Ahora entiendes las bases para todos los Quests!")

if __name__ == "__main__":
    main()