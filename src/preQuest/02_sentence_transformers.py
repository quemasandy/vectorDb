#!/usr/bin/env python3
"""
Pre-Quest Example 2: Sentence Transformers
Aprende cómo usar modelos pre-entrenados para convertir texto a vectores
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def verificar_instalacion():
    """
    Verificar si sentence-transformers está instalado
    """
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        print("❌ sentence-transformers no está instalado")
        print("📥 Instalar con: pip install sentence-transformers")
        return False

def ejemplo_embeddings_basicos():
    """
    🎯 Objetivo: Generar embeddings con sentence-transformers
    """
    if not verificar_instalacion():
        return None
    
    from sentence_transformers import SentenceTransformer
    
    print("🤖 EJEMPLO 1: Embeddings con Sentence Transformers")
    print("=" * 60)
    
    # Cargar modelo pre-entrenado
    print("📥 Cargando modelo 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Modelo cargado exitosamente")
    
    # Frases de ejemplo
    frases = [
        "El gato está durmiendo en el sofá",
        "Un felino descansa en el mueble",
        "El perro ladra en el jardín",
        "Un canino hace ruido afuera",
        "Me gusta programar en Python",
        "Disfruto codificar con Python",
        "El clima está soleado hoy",
        "Llueve mucho esta mañana"
    ]
    
    print(f"\n📝 Procesando {len(frases)} frases...")
    
    # Generar embeddings
    embeddings = model.encode(frases)
    
    print(f"✅ Embeddings generados!")
    print(f"📏 Dimensión de cada vector: {embeddings.shape[1]}")
    print(f"🔢 Forma del array: {embeddings.shape}")
    
    # Mostrar información de cada embedding
    for i, (frase, embedding) in enumerate(zip(frases, embeddings)):
        print(f"\n🔤 Frase {i+1}: '{frase}'")
        print(f"🔢 Vector (primeros 5 valores): {embedding[:5]}")
        print(f"📊 Magnitud del vector: {np.linalg.norm(embedding):.3f}")
    
    return frases, embeddings, model

def calcular_similitudes_embeddings(frases, embeddings):
    """
    🎯 Objetivo: Calcular similitudes entre embeddings
    """
    print("\n\n🔍 EJEMPLO 2: Similitudes entre Embeddings")
    print("=" * 60)
    
    # Calcular matriz de similitudes
    similitudes = cosine_similarity(embeddings)
    
    print("📊 Matriz de similitudes (solo valores > 0.5):")
    print("   ", "  ".join([f"{i:2d}" for i in range(len(frases))]))
    
    for i in range(len(frases)):
        fila = f"{i:2d} "
        for j in range(len(frases)):
            if similitudes[i, j] > 0.5:
                fila += f"{similitudes[i, j]:.2f} "
            else:
                fila += "---- "
        print(fila)
    
    # Encontrar pares más similares
    print("\n🏆 Top 5 pares más similares:")
    pares_similitud = []
    
    for i in range(len(frases)):
        for j in range(i+1, len(frases)):
            pares_similitud.append((
                similitudes[i, j], 
                i, j, 
                frases[i][:30] + "...", 
                frases[j][:30] + "..."
            ))
    
    # Ordenar por similitud descendente
    pares_similitud.sort(reverse=True)
    
    for idx, (sim, i, j, frase1, frase2) in enumerate(pares_similitud[:5]):
        print(f"{idx+1}. Similitud: {sim:.3f}")
        print(f"   📝 '{frase1}' ↔ '{frase2}'")

def comparar_metodos_vectorizacion():
    """
    🎯 Objetivo: Comparar diferentes métodos de vectorización
    """
    print("\n\n⚖️ EJEMPLO 3: Comparación de Métodos")
    print("=" * 60)
    
    if not verificar_instalacion():
        return
    
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Frases para comparar
    frases = [
        "gato animal doméstico",
        "felino mascota casa",
        "perro animal leal",
        "canino mascota fiel"
    ]
    
    print("📝 Frases de prueba:")
    for i, frase in enumerate(frases):
        print(f"   {i+1}. '{frase}'")
    
    # Método 1: TF-IDF (tradicional)
    print("\n📊 Método 1: TF-IDF")
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(frases).toarray()
    
    # Similitud entre frases 1 y 2 (gato vs felino)
    sim_tfidf = cosine_similarity([tfidf_vectors[0]], [tfidf_vectors[1]])[0][0]
    print(f"   Similitud 'gato animal' vs 'felino mascota': {sim_tfidf:.3f}")
    
    # Método 2: Sentence Transformers
    print("\n🤖 Método 2: Sentence Transformers")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st_vectors = model.encode(frases)
    
    sim_st = cosine_similarity([st_vectors[0]], [st_vectors[1]])[0][0]
    print(f"   Similitud 'gato animal' vs 'felino mascota': {sim_st:.3f}")
    
    print("\n💡 Observaciones:")
    print(f"   • TF-IDF: {sim_tfidf:.3f} (basado en palabras exactas)")
    print(f"   • Sentence Transformers: {sim_st:.3f} (entiende significado)")
    print("   • Sentence Transformers captura mejor la semántica!")

def busqueda_semantica_ejemplo():
    """
    🎯 Objetivo: Demostrar búsqueda semántica
    """
    print("\n\n🔍 EJEMPLO 4: Búsqueda Semántica")
    print("=" * 60)
    
    if not verificar_instalacion():
        return
    
    from sentence_transformers import SentenceTransformer
    
    # Base de datos de documentos
    documentos = [
        "Python es un lenguaje de programación versátil y fácil de aprender",
        "La inteligencia artificial está revolucionando la tecnología",
        "Los gatos son animales independientes y cariñosos",
        "Machine learning utiliza algoritmos para aprender de datos",
        "Los perros son leales compañeros de los humanos",
        "JavaScript es popular para desarrollo web",
        "Las redes neuronales imitan el funcionamiento del cerebro",
        "Los felinos son cazadores naturales muy eficientes",
        "React es una biblioteca de JavaScript para interfaces",
        "El deep learning es una rama del machine learning"
    ]
    
    print(f"📚 Base de datos: {len(documentos)} documentos")
    
    # Cargar modelo y generar embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(documentos)
    
    # Consultas de prueba
    consultas = [
        "programación y código",
        "animales domésticos",
        "inteligencia artificial"
    ]
    
    for consulta in consultas:
        print(f"\n🔎 Consulta: '{consulta}'")
        
        # Generar embedding de la consulta
        consulta_embedding = model.encode([consulta])
        
        # Calcular similitudes
        similitudes = cosine_similarity(consulta_embedding, doc_embeddings)[0]
        
        # Encontrar top 3 más similares
        indices_top = np.argsort(similitudes)[::-1][:3]
        
        print("   📊 Top 3 resultados:")
        for i, idx in enumerate(indices_top):
            print(f"   {i+1}. Similitud: {similitudes[idx]:.3f}")
            print(f"      📄 '{documentos[idx]}'")

def visualizar_embeddings():
    """
    🎯 Objetivo: Visualizar embeddings en 2D
    """
    print("\n\n📈 EJEMPLO 5: Visualización de Embeddings")
    print("=" * 60)
    
    if not verificar_instalacion():
        return
    
    from sentence_transformers import SentenceTransformer
    
    # Frases agrupadas por temas
    frases = [
        # Tecnología
        "programación en Python",
        "desarrollo de software", 
        "inteligencia artificial",
        # Animales
        "gatos domésticos",
        "perros leales",
        "animales de compañía",
        # Comida
        "pizza italiana deliciosa",
        "pasta con salsa",
        "comida mediterránea",
        # Deportes
        "fútbol y goles",
        "baloncesto profesional",
        "deportes de equipo"
    ]
    
    categorias = ["tech", "tech", "tech", "animals", "animals", "animals", 
                 "food", "food", "food", "sports", "sports", "sports"]
    
    # Generar embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(frases)
    
    # Reducir dimensionalidad con PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    colores = {'tech': 'red', 'animals': 'blue', 'food': 'green', 'sports': 'orange'}
    
    for categoria in set(categorias):
        indices = [i for i, cat in enumerate(categorias) if cat == categoria]
        x = [embeddings_2d[i][0] for i in indices]
        y = [embeddings_2d[i][1] for i in indices]
        plt.scatter(x, y, c=colores[categoria], label=categoria, s=100, alpha=0.7)
        
        # Añadir etiquetas
        for i in indices:
            plt.annotate(frases[i][:15] + "...", 
                        (embeddings_2d[i][0], embeddings_2d[i][1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Visualización de Embeddings por Categorías')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    try:
        plt.tight_layout()
        plt.savefig('/home/andy/quicksight/vectorDb/src/preQuest/embeddings_visualization.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        print("💾 Gráfico guardado como 'embeddings_visualization.png'")
    except Exception as e:
        print(f"⚠️ No se pudo crear el gráfico: {e}")

def main():
    """
    Ejecutar todos los ejemplos
    """
    print("🤖 PRE-QUEST: Sentence Transformers")
    print("🎯 Embeddings profesionales con modelos pre-entrenados")
    print("=" * 80)
    
    # Verificar instalación
    if not verificar_instalacion():
        print("\n💡 Para instalar sentence-transformers:")
        print("   pip install sentence-transformers")
        print("\n🎮 Mientras tanto, puedes ejecutar 01_manual_vectors.py")
        return
    
    # Ejecutar ejemplos
    frases, embeddings, model = ejemplo_embeddings_basicos()
    if embeddings is not None:
        calcular_similitudes_embeddings(frases, embeddings)
        comparar_metodos_vectorizacion()
        busqueda_semantica_ejemplo()
        
        try:
            visualizar_embeddings()
        except Exception as e:
            print(f"⚠️ Error en visualización: {e}")
    
    print("\n\n🎉 ¡Ejemplos de Sentence Transformers completados!")
    print("💡 Conceptos clave aprendidos:")
    print("   • Modelos pre-entrenados para embeddings")
    print("   • Embeddings densos vs sparse")
    print("   • Similitud semántica vs similitud léxica")
    print("   • Búsqueda semántica en documentos")
    print("   • Visualización de espacios vectoriales")
    print("\n🎮 ¡Ahora tienes las bases para Quest 2: Real Vector Database!")

if __name__ == "__main__":
    main()