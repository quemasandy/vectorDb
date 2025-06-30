#!/usr/bin/env python3
"""
Pre-Quest Example 1: Manual Text Vectorization
Aprende cómo convertir texto a vectores manualmente
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def ejemplo_vectores_manuales():
    """
    🎯 Objetivo: Entender cómo se puede representar texto como números
    """
    print("📝 EJEMPLO 1: Vectores One-Hot (Presencia/Ausencia)")
    print("=" * 60)
    
    # Frases de ejemplo
    frases = [
        "el gato come pescado",
        "el perro come carne", 
        "el gato duerme mucho",
        "el perro ladra fuerte"
    ]
    
    # Crear vocabulario (todas las palabras únicas)
    vocabulario = set()
    for frase in frases:
        palabras = frase.lower().split()
        vocabulario.update(palabras)
    
    vocabulario = sorted(list(vocabulario))
    print(f"📚 Vocabulario: {vocabulario}")
    print(f"📏 Tamaño del vocabulario: {len(vocabulario)}")
    
    # Crear vectores one-hot
    vectores = []
    for i, frase in enumerate(frases):
        palabras = frase.lower().split()
        vector = [1 if palabra in palabras else 0 for palabra in vocabulario]
        vectores.append(vector)
        
        print(f"\n🔤 Frase {i+1}: '{frase}'")
        print(f"🔢 Vector: {vector}")
    
    return vocabulario, vectores

def ejemplo_vectores_frecuencia():
    """
    🎯 Objetivo: Usar frecuencia de palabras en lugar de solo presencia
    """
    print("\n\n📊 EJEMPLO 2: Vectores TF (Term Frequency)")
    print("=" * 60)
    
    frases = [
        "gato gato come",
        "perro come come carne",
        "gato duerme",
        "perro perro ladra"
    ]
    
    # Crear vocabulario
    vocabulario = set()
    for frase in frases:
        palabras = frase.lower().split()
        vocabulario.update(palabras)
    
    vocabulario = sorted(list(vocabulario))
    print(f"📚 Vocabulario: {vocabulario}")
    
    # Crear vectores de frecuencia
    vectores = []
    for i, frase in enumerate(frases):
        palabras = frase.lower().split()
        contador = Counter(palabras)
        vector = [contador.get(palabra, 0) for palabra in vocabulario]
        vectores.append(vector)
        
        print(f"\n🔤 Frase {i+1}: '{frase}'")
        print(f"🔢 Vector: {vector}")
        print(f"📈 Palabras contadas: {dict(contador)}")
    
    return vocabulario, vectores

def ejemplo_vectores_caracteristicas():
    """
    🎯 Objetivo: Crear vectores basados en características semánticas
    """
    print("\n\n🎭 EJEMPLO 3: Vectores de Características Semánticas")
    print("=" * 60)
    
    # Definir características
    caracteristicas = [
        "es_animal", "es_domestico", "come_carne", "hace_ruido", "es_pequeno"
    ]
    
    # Palabras y sus características
    palabras_caracteristicas = {
        "gato": [1, 1, 1, 1, 1],      # animal, doméstico, carnívoro, hace ruido, pequeño
        "perro": [1, 1, 1, 1, 0],     # animal, doméstico, carnívoro, hace ruido, no tan pequeño
        "león": [1, 0, 1, 1, 0],      # animal, no doméstico, carnívoro, hace ruido, no pequeño
        "mesa": [0, 1, 0, 0, 0],      # no animal, doméstico (en casa), no come, no hace ruido, variable
        "ratón": [1, 0, 0, 1, 1],     # animal, no doméstico, no carnívoro, hace ruido, pequeño
    }
    
    print(f"🏷️ Características: {caracteristicas}")
    print("\n📋 Vectores por palabra:")
    
    for palabra, vector in palabras_caracteristicas.items():
        print(f"🔤 {palabra:6} → {vector}")
        
        # Explicar cada característica
        caracteristicas_activas = [
            caracteristicas[i] for i, val in enumerate(vector) if val == 1
        ]
        print(f"   ✅ Características: {', '.join(caracteristicas_activas)}")
    
    return caracteristicas, palabras_caracteristicas

def calcular_similitud(vector1, vector2, nombre1, nombre2):
    """
    Calcular similitud coseno entre dos vectores
    """
    # Convertir a arrays de numpy
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Calcular similitud coseno
    producto_punto = np.dot(v1, v2)
    magnitud1 = np.linalg.norm(v1)
    magnitud2 = np.linalg.norm(v2)
    
    if magnitud1 == 0 or magnitud2 == 0:
        similitud = 0
    else:
        similitud = producto_punto / (magnitud1 * magnitud2)
    
    print(f"🔗 Similitud entre '{nombre1}' y '{nombre2}': {similitud:.3f}")
    return similitud

def ejemplo_similitudes():
    """
    🎯 Objetivo: Comparar similitudes entre diferentes vectores
    """
    print("\n\n🔍 EJEMPLO 4: Calculando Similitudes")
    print("=" * 60)
    
    # Usar vectores de características del ejemplo anterior
    caracteristicas, palabras_vectores = ejemplo_vectores_caracteristicas()
    
    # Comparar algunas palabras
    palabras = list(palabras_vectores.keys())
    
    print("\n📊 Comparaciones de similitud:")
    
    # Comparar gato vs perro
    calcular_similitud(
        palabras_vectores["gato"], 
        palabras_vectores["perro"],
        "gato", "perro"
    )
    
    # Comparar gato vs león
    calcular_similitud(
        palabras_vectores["gato"], 
        palabras_vectores["león"],
        "gato", "león"
    )
    
    # Comparar gato vs mesa
    calcular_similitud(
        palabras_vectores["gato"], 
        palabras_vectores["mesa"],
        "gato", "mesa"
    )
    
    # Comparar perro vs león
    calcular_similitud(
        palabras_vectores["perro"], 
        palabras_vectores["león"],
        "perro", "león"
    )

def visualizar_vectores():
    """
    🎯 Objetivo: Visualizar vectores en un gráfico
    """
    print("\n\n📈 EJEMPLO 5: Visualización de Vectores")
    print("=" * 60)
    
    # Vectores de ejemplo (reducidos a 2D para visualización)
    animales = {
        "gato": [0.8, 0.3],      # [domestico, tamaño]
        "perro": [0.9, 0.7],
        "león": [0.1, 0.9],
        "ratón": [0.2, 0.1],
        "elefante": [0.0, 1.0]
    }
    
    # Crear el gráfico
    plt.figure(figsize=(10, 8))
    
    for animal, (x, y) in animales.items():
        plt.scatter(x, y, s=100, alpha=0.7)
        plt.annotate(animal, (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Nivel Doméstico (0=salvaje, 1=doméstico)')
    plt.ylabel('Tamaño (0=pequeño, 1=grande)')
    plt.title('Vectores de Animales en Espacio 2D')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Guardar y mostrar
    plt.tight_layout()
    plt.savefig('/home/andy/quicksight/vectorDb/src/preQuest/vectores_animales.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("💾 Gráfico guardado como 'vectores_animales.png'")

def main():
    """
    Ejecutar todos los ejemplos
    """
    print("🚀 PRE-QUEST: Transformando Texto a Vectores")
    print("🎯 Aprende los fundamentos antes de empezar Quest 1")
    print("=" * 80)
    
    # Ejecutar ejemplos
    ejemplo_vectores_manuales()
    ejemplo_vectores_frecuencia()
    ejemplo_vectores_caracteristicas()
    ejemplo_similitudes()
    
    try:
        visualizar_vectores()
    except Exception as e:
        print(f"⚠️ No se pudo crear el gráfico: {e}")
        print("   (Esto es normal si no tienes entorno gráfico)")
    
    print("\n\n🎉 ¡Ejemplos completados!")
    print("💡 Conceptos clave aprendidos:")
    print("   • Vectores One-Hot (presencia/ausencia)")
    print("   • Vectores de Frecuencia (TF)")
    print("   • Vectores de Características Semánticas")
    print("   • Cálculo de Similitud Coseno")
    print("   • Visualización de Vectores")
    print("\n🎮 ¡Ahora estás listo para Quest 1: Vector Playground!")

if __name__ == "__main__":
    main()