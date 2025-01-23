#import "../template.typ": conf

#set document(
	title: "Anteproyecto: Descripción de Imágenes con Mecanismo de Atención",
	author: "Micaela Del Longo",
	date: datetime(year: 2025, month: 1, day: 23)
)

#show: conf.with(
	title: [#context(document.title)],
	author: [#context(document.author.first())],
	date: [#context(document.date.display("[day] de Enero de [year]"))],
	university: [Universidad Nacional de Cuyo],
	college: [Facultad de Ingeniería],
	career: [Licenciatura en Ciencias de la Computación],
	course: [Inteligencia Artificial II],
	professors: ([Dr. Rodrigo Gonzalez], [Dr. Jorge Guerra]),
)

#outline(
	title: "Contenidos",
	indent: auto
)

= Introducción

En la era de los datos visuales, las imágenes han tomado un rol fundamental en la comunicación y el intercambio de información @vox-new-era. Un modelo de descripción automática de imágenes (Image Captioning) @sci-dir-image-capt combina procesamiento de imágenes y lenguaje natural para generar textos que describan contenido visual. Este tipo de modelos tiene aplicaciones diversas, como en tecnologías de accesibilidad para personas con discapacidades visuales, organización automática de imágenes en plataformas digitales, y mejora de la búsqueda basada en imágenes.

El presente anteproyecto propone el desarrollo de un modelo que combine Redes Neuronales Convolucionales (CNN) para la extracción de características de imágenes y Redes Neuronales Recurrentes (RNN) para la generación de texto que describa el contenido visual. Además, se incluirá un mecanismo de atención que permita al modelo enfocar regiones específicas de la imagen al generar cada palabra @tensor-image-capt. Inicialmente, se desarrollará una versión del modelo sin mecanismo de atención, la cual se utilizará como base para comparar los resultados obtenidos al implementar la versión con atención.

= Descripción del Problema

El problema se enmarca en la categoría de aprendizaje supervisado @islp[c. ~12] y clasificación secuencial @islp[c. ~4]. Dado un conjunto de datos etiquetados con imágenes y sus respectivas descripciones textuales, el modelo debe aprender a asociar las características visuales de una imagen con las palabras y frases que describen su contenido.

= Objetivos

Como se describió anteriormente, el objetivo principal de este proyecto es desarrollar un modelo basado en CNNs y RNNs que genere descripciones textuales precisas y coherentes para una imagen dada.

Para lograr este objetivo, se plantean los siguientes objetivos específicos:

- Extraer características visuales de las imágenes utilizando una CNN preentrenada (ResNet50 @resnet@resnet-paper).

- Implementar una RNN (LSTM @deep-learn[p. ~297]) para generar descripciones secuenciales de texto basadas en las características visuales.

- Crear un mecanismo de atención @deep-learn[c. ~11.4] para permitir al modelo enfocar diferentes regiones de la imagen durante la generación de texto.

- Probar los modelos con distintas cantidades de capas y neuronas para evaluar su impacto en el desempeño.

- Evaluar los modelos utilizando métricas de evaluación de lenguaje natural como BLEU @BLEU, METEOR @METEOR y/o CIDEr @CIDEr.

- Comparar los resultados obtenidos con y sin mecanismo de atención para analizar la mejora en la calidad de las descripciones generadas.

Además, se plantean objetivos adicionales: implementar beam search @beam para mejorar la generación de texto y realizar fine-tuning de ResNet @deep-learn[p. ~234] para optimizar la extracción de características visuales según el dominio del conjunto de datos. Dichos objetivos se abordarán en caso de cumplir con los objetivos principales en tiempo y forma.

= Datos de Entrada

Los datos de entrada al modelo consistirán en:

- _Imágenes:_ Fotografías en formato RGB, que serán procesadas a través de una CNN para extraer mapas de características.

- _Descripciones textuales:_ Secuencias de palabras en formato texto que describen el contenido visual de cada imagen.

Como posible fuente de datos, se considera el dataset Flickr8K @flickr8k o Flickr30K @flickr30k, que contienen imágenes de _Flickr_ con descripciones en inglés. También, se considera el conjunto de datos MS COCO @mscoco, que contiene imágenes de escenas cotidianas y sus descripciones asociadas.

= Algoritmos a Implementar

El modelo implementará una arquitectura que combina diferentes técnicas:

- _Redes Neuronales Convolucionales (CNNs):_ ResNet50, previamente entrenada, para extraer representaciones de alto nivel de las imágenes.

- _Redes Neuronales Recurrentes (RNNs):_ Una LSTM (Long Short-Term Memory) que tomará como entrada el vector de características de la imagen y generará las palabras de la descripción de forma secuencial.

- _Mecanismo de atención:_ Una capa que calculará pesos de atención para cada región de la imagen, permitiendo que el modelo enfoque diferentes partes del mapa de características mientras genera cada palabra. Este mecanismo será implementado tras evaluar la versión inicial sin atención.

= Descripción del Sistema

== Preprocesamiento de Datos

En el caso de las _imágenes_, estás serán redimensionadas y normalizadas para cumplir con los requisitos de entrada de ResNet50. Posteriormente, se extraerán características de las capas intermedias de ResNet para obtener un mapa de características de alto nivel.

Mientras que las _descripciones textuales_ serán tokenizadas, y se convertirá cada palabra en un índice utilizando un vocabulario creado a partir del conjunto de datos. Se aplicará padding para unificar la longitud de las secuencias de texto.

== Arquitectura del Modelo

El modelo constará de las siguientes capas y componentes:

1. Extracción de características visuales:
	- Se utilizará una ResNet50 preentrenada en ImageNet.
	- El mapa de características extraído tendrá dimensiones fijas, representando las características visuales de diferentes regiones de la imagen.
2. Codificación del texto:
	- Una capa de embedding para convertir las palabras en vectores densos.
	- Una LSTM para procesar las secuencias de palabras generadas hasta el momento.
3. Mecanismo de atención (en la versión avanzada):
	- Una capa de atención que calculará pesos para cada región del mapa de características basado en la etapa actual de la LSTM.
	- Los pesos se combinarán con las características visuales para producir un vector de contexto.
4. Generación de texto:
	- La LSTM combinará el vector de contexto (características visuales ponderadas) con el estado actual para predecir la siguiente palabra.
	- Una capa densa con softmax se utilizará para predecir la probabilidad de cada palabra en el vocabulario.

== Flujo de Trabajo

El flujo de trabajo del sistema se divide en dos etapas principales: entrenamiento e inferencia.

Durante el _entrenamiento_, las imágenes y las descripciones tokenizadas se pasan al modelo. Se calcula la pérdida basada en la predicción de palabras del modelo y las palabras reales de la descripción. El optimizador ajusta los pesos de las redes para minimizar esta pérdida.

En la etapa de _inferencia_, una imagen se pasa por ResNet para obtener características visuales. Luego, la LSTM y el mecanismo de atención (si está implementado) generan una descripción palabra por palabra. Si se implementa beam search, se generarán múltiples descripciones candidatas, y se seleccionará la más probable.

#bibliography(
	"bib.yml",
	full: true
)
