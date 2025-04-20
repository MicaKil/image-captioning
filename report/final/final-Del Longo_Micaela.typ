#import "../template.typ": conf, sub_caption, appendix

#set document(
	title: "Trabajo Final: Descripción de Imágenes con Redes Neuronales",
	author: "Micaela Del Longo",
	date: datetime(year: 2025, month: 4, day: 23)
)

#show: conf.with(
	title: [#context(document.title)],
	author: [#context(document.author.first())],
	date: [#context(document.date.display("[day] de abril de [year]"))],
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

El objetivo principal de este informe es el de explorar, desarrollar y evaluar diferentes arquitecturas de modelos de inteligencia artificial para la tarea de descripción automática de imágenes (image captioning@sci-dir-image-capt). Para ello, se aborda el proceso completo, comenzando por un análisis exploratorio y preprocesamiento de los datos. Se definen las métricas de evaluación, como BLEU@BLEU y CIDEr@CIDEr, que permiten cuantificar la calidad de las descripciones generadas en comparación con las referencias humanas.

En el núcleo de este trabajo se encuentra la implementación y comparación de distintas arquitecturas de modelos. Se exploran enfoques clásicos basados en la combinación de redes convolucionales (ResNet50@resnet-paper) como codificadores de imágenes y redes recurrentes (LSTM) o mecanismos de atención como decodificadores de texto. Además, se investiga el uso de arquitecturas más recientes basadas en Transformers, como el codificador Swin@swin, para evaluar su impacto en el rendimiento.

Los modelos desarrollados fueron entrenados y evaluados utilizando dos conjuntos de datos estándar en este campo: Flickr8k@flickr8k y COCO@coco. Los experimentos realizados permiten comparar el desempeño de las distintas arquitecturas bajo métricas objetivas. Adicionalmente, se incluye un análisis de interpretabilidad mediante la visualización de mapas de atención, buscando entender qué regiones de la imagen son más relevantes para el modelo al generar palabras específicas de la descripción.

Este informe está estructurado de la siguiente manera: La @métodos detalla en profundidad los métodos empleados, incluyendo el tratamiento de los datos, las métricas de evaluación, las arquitecturas de los modelos, el proceso de entrenamiento y las herramientas utilizadas. La @exp presenta los experimentos realizados y discute los resultados obtenidos en los datasets Flickr8k y COCO, incluyendo el análisis de interpretabilidad. Finalmente, la @conclusiones resume las conclusiones principales del estudio y propone posibles líneas de trabajo futuro.

= Métodos <métodos>

== Análisis Exploratorio de los Datos

Para el entrenamiento y la evaluación de los modelos de descripción de imágenes, se utilizaron dos datasets estándar: Flickr8k@flickr8k y COCO (Common Objects in Context)@coco. Ambos datasets consisten en imágenes acompañadas de descripciones textuales en inglés, proporcionando una base para el aprendizaje supervisado.

El dataset Flickr8k contiene 8091 imágenes únicas obtenidas del sitio web Flickr. Cada imagen está asociada con cinco descripciones distintas, lo que resulta en un total de 40457 pares imagen-descripción. Dada su escala moderada, Flickr8k es comúnmente utilizado para desarrollos iniciales y pruebas rápidas de modelos.

Por otro lado, COCO es un dataset a gran escala que incluye 123287 imágenes de escenas complejas y cotidianas. Cada imagen cuenta con al menos cinco descripciones, sumando un total de 616767 pares imagen-descripción. Su tamaño y diversidad lo convierten en un benchmark para evaluar la capacidad de generalización de los modelos.

Los datasets se dividieron en conjuntos de entrenamiento (train), validación (validation) y prueba (test). En el caso de Flickr8k, se aplicó una división personalizada del 80% para entrenamiento, 10% para validación y 10% para prueba. Para COCO, se mantuvo la división oficial preestablecida, que corresponde aproximadamente a una distribución del 75%, 15% y 15% respectivamente.

En ambos datasets se garantizó que no hubiera solapamiento de imágenes entre los diferentes conjuntos; es decir, una imagen presente en el conjunto de entrenamiento no podía aparecer ni en el de validación ni en el de prueba, asegurando así que la evaluación midiera la capacidad del modelo para generalizar a imágenes no vistas.

#figure(
  table(
    columns: 3,
    [Conjunto],[Porcentaje],[Total de Pares],
    [_train_],[80],[32360],
    [_validation_],[10],[4046],
    [_test_],[10],[4051],
  ) , 
  caption: [ Distribución de los conjuntos train, validation y test en el dataset Flickr8k.],
  placement: bottom
) <f8k-dist>

#figure(
  table(
    columns: 3,
    [Conjunto],[Porcentaje],[Total de Pares],
    [_train_],[75],[414113],
    [_validation_],[15],[101327],
    [_test_],[15],[101327],
  ) , 
  caption: [Distribución de los conjuntos train, validation y test en el dataset COCO.],
  placement: bottom
) <coco-dist>

Con estos conjuntos se realizó un análisis de las descripciones textuales para comprender sus características principales, como la longitud y el vocabulario utilizado.
    
En Flickr8k, la longitud promedio de las descripciones es notablemente consistente a través de los conjuntos de entrenamiento, validación y prueba, situándose alrededor de 11.8 palabras, con una desviación estándar de aproximadamente 3.9 palabras. Las longitudes varían desde 1 hasta 38 palabras. Los histogramas y diagramas de caja (@hist-words-f8k[Figuras] @box-words-f8k[ y]) ilustran esta distribución, mostrando una concentración alrededor de la media y una cola derecha debido a descripciones más largas.

#figure(
image("eda/histogram_caption_len_flickr8k.png"),
  caption: [Histograma de frecuencia del largo descripción de imagen en cada set del dataset Flickr8k.],
  placement: auto
) <hist-words-f8k>

#figure(
image("eda/boxplot_caption_len_flickr8k.png", height: 13cm),
  caption: [Diagramas de caja y bigotes para el largo descripción de imagen en cada set del dataset Flickr8k.],
  placement: auto
) <box-words-f8k>

Para el dataset COCO, la longitud promedio de las descripciones es ligeramente menor, alrededor de 11.3 palabras, con una desviación estándar más baja (aproximadamente 2.6 palabras), lo que sugiere una menor variabilidad en la longitud de las frases en comparación con Flickr8k. El rango de longitud observado va desde 6 hasta 57 palabras. Las distribuciones (@hist-words-coco[Figuras] @box-words-coco[ y]) también muestran una concentración alrededor de la media con una cola derecha.

#figure(
  image("eda/histogram_caption_len_coco.png"),
  caption: [Histograma de frecuencia de largo descripción de imagen en cada set del dataset COCO.],
  placement: auto
)<hist-words-coco>

#figure(
  image("eda/boxplot_caption_len_coco.png", height: 13cm),
  caption: [Diagramas de caja y bigotes para el largo descripción de imagen en cada set del dataset COCO.],
  placement: auto
)<box-words-coco>

El análisis del vocabulario revela las diferencias entre los dos datasets. Flickr8k (conjunto de entrenamiento) contiene un total de 201387 palabras, con 7987 palabras únicas. Las palabras más frecuentes (excluyendo stopwords comunes) incluyen términos relacionados con personas y animales ("man", "dog", "boy", "woman", "girl", "people"), colores ("white", "black", "red", "brown", "blue"), acciones ("wearing", "running", "playing", "standing", "jumping", "sitting") y objetos comunes ("water", "ball", "shirt", "grass", "snow").

#figure(
  image("eda/flickr8k_vocab_analysis.png"),
  caption: [Las 30 palabras, bigramas y trigramas más frecuentes en el conjunto de entrenamiento de Flickr8k.],
) <words-f8k>

COCO (conjunto de entrenamiento) presenta un vocabulario mucho más extenso, con un total de 2397585 palabras y 24806 palabras únicas. Las palabras más frecuentes reflejan la naturaleza diversa de las escenas, incluyendo términos relacionados con personas ("man", "people", "woman", "person", "group", "young"), ubicaciones y objetos comunes ("sitting", "standing", "table", "street", "field", "room", "train", "plate", "cat", "dog", "water"), atributos ("white", "large", "small", "black", "red") y acciones o relaciones espaciales ("next", "holding", "top", "near", "front", "riding").

#figure(
  image("eda/coco_vocab_analysis.png"),
  caption: [Las 30 palabras, bigramas y trigramas más frecuentes en el conjunto de entrenamiento de COCO.],
  placement: auto
) <words-coco>

Las @words-f8k[Figuras] @words-coco[ y] muestran las palabras, bigramas y trigramas más frecuentes para cada dataset, ofreciendo una visión más detallada de los patrones lingüísticos predominantes.

=== Preprocesamiento de los Datos

Para estandarizar el texto de las descripciones, a este se lo pasó a minúscula y se convirtieron los caracteres a ASCII en caso de no estarlo. Como estrategias de tokenización se utilizó tokenización por palabras y Byte-Pair Encoding (BPE). 

Para crear los vocabularios basados en palabras se tomó como frecuencia mínima de palabra 3 y 5, en los datasets de Flickr8K y COCO, respectivamente. Mientras que para los vocabularios basados en BPE, se tomaron los primeros 3500 y 8500 tokens más frecuentes en los datasets de Flickr8K y COCO, respectivamente.

El vocabulario generado cuenta con los siguientes tokens especiales:

- `SOS`: Start of Sentence (Inicio de oración).
- `EOS`: End of Sentence (Fin de oración).
- `PAD`: Padding (Relleno).
- `UNK`: Unknown Desconocido (no pertenece al vocabulario).

Para crear los vocabularios en ambos casos solo se utilizó el dataset de entrenamiento para evitar exponer el modelo a datos no vistos durante el entrenamiento, inflando artificialmente las métricas de rendimiento y afectando la integridad de la evaluación.

En el caso de las imágenes, estas fueron primero redimensionadas a `224x224` o a `256x256` y luego fueron normalizadas con una media de `[0.485, 0.456, 0.406]` y una desviación estándar igual a `[0.229, 0.224, 0.225]`.

== Métricas

Para evaluar cuantitativamente el rendimiento de los modelos de generación de descripciones de imágenes, se emplearon métricas estándar en el campo: BLEU y CIDEr. Estas métricas comparan las descripciones generadas automáticamente con un conjunto de descripciones de referencia creadas por humanos.

=== BLEU (Bilingual Evaluation Understudy)


La métrica BLEU@BLEU, concebida originalmente para la evaluación de traducciones automáticas, mide la similitud léxica entre una descripción generada ($hat(y)$) y un conjunto de descripciones de referencia $S_i = (y^((i, 1)), dots, y^((i, N_i)))$ mediante el análisis de la coincidencia de n-gramas. Se calculan puntuaciones para diferentes longitudes de n-gramas (e.g., BLEU-1 para unigramas, BLEU-4 para tetragramas).

La puntuación final combina la precisión modificada de los n-gramas ($p_n$) con un factor de penalización por brevedad (Brevity Penalty, BP) para evitar que descripciones excesivamente cortas obtengan puntuaciones artificialmente altas. La fórmula general para un corpus $hat(S)=(hat(y)^((1)), dots, hat(y)^((M))$ con referencias $S=(S_1, dots, S_M)$ es:

#figure(
  $"BLEU" = "BP" dot exp(sum_(n=1)^N w_n log p_n)$
)

Donde:

- $p_n$ es la precisión modificada de los n-gramas de orden n. Se calcula como la suma de los recuentos de n-gramas candidatos que coinciden con alguna referencia (recortados al máximo número de veces que aparecen en una referencia), dividida por el número total de n-gramas en la descripción candidata.

- $w_n$ son pesos positivos que suman 1 (usualmente $w_n = 1/N$, dando igual importancia a cada orden de n-grama hasta N).

- $"BP"$ es la penalización por brevedad, calculada como:
  $"BP" = {1 "si" c > r, e^{(1 - r/c)} & "si" c <= r}$
  donde $c$ es la longitud total del corpus candidato y $r$ es la suma de las longitudes de referencia efectivas (la longitud de referencia más cercana a la longitud candidata para cada oración).

La puntuación BLEU varía entre 0 y 1, donde valores más cercanos a 1 indican una mayor similitud léxica con las referencias. Aunque es computacionalmente eficiente y ampliamente adoptada, BLEU presenta limitaciones, como su insensibilidad al significado semántico y a la estructura gramatical, lo que puede llevar a evaluaciones imprecisas si la descripción generada es conceptualmente correcta pero utiliza sinónimos o una sintaxis diferente a las referencias.

=== CIDEr (Consensus-based Image Description Evaluation)

CIDEr@CIDEr es una métrica desarrollada específicamente para la evaluación de descripciones de imágenes, diseñada para medir el consenso entre la descripción generada y el conjunto de referencias humanas. A diferencia de BLEU, CIDEr incorpora una perspectiva semántica al ponderar la importancia de los n-gramas mediante la estrategia TF-IDF (Term Frequency-Inverse Document Frequency). Esta técnica asigna mayor peso a los n-gramas que son frecuentes en una descripción particular pero infrecuentes en el conjunto total de descripciones de referencia, capturando así términos más distintivos y descriptivos del contenido visual.

Para cada n-grama ($w_k$) de longitud n, se calcula su vector TF-IDF $g_n (hat(y))$ para la descripción candidata $hat(y)$ y $g_n (y^((i,j)))$ para cada referencia $y^((i,j)) in S_i$. La puntuación $"CIDEr"_n$ para n-gramas de longitud n se calcula como el promedio de la similitud coseno entre el vector TF-IDF de la candidata y los vectores TF-IDF de las referencias:

#figure(
  $"CIDEr"_n(hat(y), S_i) = frac(1, N_i) sum_(j=1)^(N_i) frac(g_n (hat(y)) dot g_n (y^((i,j))),|g_n (hat(y))| |g_n (y^((i,j)))|)$
)

La puntuación CIDEr final se obtiene promediando las puntuaciones $"CIDEr"_n$ para diferentes longitudes de $n$ (usualmente $n=1$ a $N=4$):

#figure(
$"CIDEr"_n(hat(y), S_i) = sum_(n=1)^N w_n "CIDEr"_n(hat(y), S_i)$
)

Donde $w_n$ son pesos (comúnmente $w_n = 1/N$). La puntuación CIDEr para un corpus se promedia sobre todas las imágenes.

CIDEr es computacionalmente más intensiva que BLEU, pero su enfoque en la relevancia semántica y el consenso la hace más correlacionada con el juicio humano sobre la calidad de las descripciones de imágenes. Es frecuentemente utilizada como función de recompensa en enfoques de aprendizaje por refuerzo para optimizar modelos de captioning. Sus valores teóricamente no están acotados superiormente, pero en la práctica pueden superar el valor de 1, indicando una fuerte correlación con las referencias consensuadas; valores más altos siempre indican mejor rendimiento.

== Arquitecturas

Se implementaron dos arquitecturas _principales_ para generar descripciones a partir de imágenes: un enfoque basado en LSTM y otro en Transformer. Ambos modelos siguen una estructura Encoder-Decoder (codificador-decodificador), ya que el proceso se divide conceptualmente en dos etapas principales: la codificación de la imagen de entrada en una representación latente y la decodificación de esta representación para generar la secuencia de palabras que constituye la descripción.

=== Modelo ResNet50-LSTM

Como su nombre lo indica, este modelo combina un encoder visual basado en la arquitectura ResNet50@resnet-paper con un decodificador LSTM para generar secuencias de texto. El componente Encoder tiene como función la extracción de características semánticas relevantes de la imagen de entrada. Para ello, como se mencionó anteriormente, se utiliza una red convolucional profunda, ResNet50, preentrenada sobre el dataset ImageNet. La utilización de pesos preentrenados permite transferir el conocimiento adquirido durante la clasificación de millones de imágenes a la tarea específica de descripción de imágenes. 

La arquitectura ResNet50 original es modificada de la siguiente forma: se suprime la capa final completamente conectada (fully connected), diseñada originalmente para la clasificación en 1000 categorías, conservando las capas convolucionales precedentes que actúan como extractoras de características. La salida de estas capas, un tensor de alta dimensionalidad (2048 características --features--), es posteriormente aplanada y procesada por una capa de proyección lineal. Esta capa adicional, compuesta por una transformación lineal, una función de activación no lineal ReLU, y una capa de regularización dropout, mapea el vector de características de alta dimensión a un espacio de embedding. Este vector resultante encapsula la información visual que será utilizada por el decodificador. 

El Encoder también cuenta con la capacidad de fine-tuning, esta funcionalidad permite ajustar el grado en que los parámetros de la ResNet50 preentrenada son modificados durante el entrenamiento del modelo de captioning. Las opciones varían desde congelar todos los pesos de (`fine_tune="none"`), permitiendo únicamente el entrenamiento de la capa de proyección y el decodificador, hasta descongelar todas las capas (`fine_tune="full"`), pasando por una opción intermedia (`fine_tune="partial"`) que actualiza únicamente las últimas capas convolucionales. 

Por último, el flujo de procesamiento (forward) consiste en pasar la imagen a través de la ResNet50 modificada, aplanar la salida y proyectarla al espacio de embedding mediante la capa lineal.

El componente Decoder se encarga de generar la secuencia textual (la descripción) a partir del vector de características proporcionado por el codificador. La arquitectura seleccionada para esta tarea es una red LSTM (Long Short-Term Memory), una variante de las redes neuronales recurrentes (RNN) especialmente diseñada para modelar dependencias a largo plazo en datos secuenciales, como es el caso del lenguaje natural. 

Un aspecto importante es la inicialización de los estados LSTM: los estados oculto (h) y de celda (c) iniciales de la LSTM no parten de cero, sino que son generados a partir del vector de características de la imagen mediante dos capas lineales dedicadas. Esta inicialización condiciona el proceso generativo al contenido visual específico de la imagen. 

El proceso de decodificación opera secuencialmente: en cada paso de tiempo, la palabra previamente generada (o un token especial de inicio SOS al principio) es convertida en un vector denso mediante una Capa de Embedding. Este embedding, junto con los estados internos de la LSTM, alimenta el módulo LSTM principal, que actualiza sus estados y produce una salida. Dicha salida es normalizada para estabilizar el aprendizaje. Finalmente, una capa lineal proyecta la salida normalizada de la LSTM al espacio del vocabulario completo, generando las puntuaciones o logits para cada palabra posible como la siguiente en la secuencia. Se aplica una estrategia para minimizar la probabilidad de generar tokens no deseados (`PAD`, `SOS`, `UNK`) asignando un sesgo fuertemente negativo a sus correspondientes logits en la capa de salida. 

El flujo de procesamiento del decodificador (forward) durante el entrenamiento recibe las características de la imagen y la secuencia completa de la descripción de referencia (con padding), procesándolos a través de los embeddings, la LSTM, la normalización y la capa lineal final para predecir los logits en cada posición de la secuencia.

El modelo final integra el Encoder y Decoder en un único módulo, definiendo la interfaz general del modelo. Su método forward orquesta el flujo de datos durante el entrenamiento, pasando la imagen por el codificador y luego las características resultantes y las descripciones de referencia por el decodificador. 

La @resnet-lstm ilustra la arquitectura del modelo y el flujo de datos a través de este. En dicho diagrama cada color de bloque indica lo siguiente:
- Blanco: Entrada
- Gris: Salida
- Verde: Capa del Encoder
- Azul: Capa de embedding
- Rojo: Capa del Decoder

#figure(
  image("figures/Resnet-LSTM.png", height: 16cm),
  caption: [Diagrama de la arquitectura y el flujo de datos en ResNet50-LSTM.],
  placement: top
) <resnet-lstm>


=== Modelo ResNet50-Attention 

Este modelo se encuentra nuevamente basado en el paradigma Encoder-Decoder, pero esta vez se hace uso extensivo de mecanismos de Atención@attn@show-attend-tell, característicos de la arquitectura Transformer. A diferencia de los enfoques basados en redes recurrentes (RNNs) como LSTM, este modelo aprovecha la capacidad de los Transformers para capturar dependencias a largo plazo y procesar información en paralelo, tanto en la codificación visual como en la decodificación textual.

El componente Encoder se basa, al igual que en la arquitectura previa, en una red convolucional profunda ResNet50 preentrenada en ImageNet para la extracción inicial de características visuales. Sin embargo, la modificación aplicada a ResNet50 es distinta: se eliminan las últimas dos capas (la capa de Average Pooling global y la capa completamente conectada --fully connected-- final), permitiendo conservar un mapa de características (`[batch_size, feature_dim, 7, 7]`). Este mapa de características retiene información sobre la disposición espacial de los elementos en la imagen, lo cual es crucial para los mecanismos de atención posteriores. A continuación, se aplica una capa de proyección convolucional, implementada mediante una convolución 1x1, seguida de una activación ReLU y Dropout espacial. Esta proyección reduce la dimensionalidad de las características al tamaño del embedding deseado, manteniendo al mismo tiempo las dimensiones espaciales (`[batch_size, embed_dim, 7, 7]`). La capacidad de realizar fine-tuning sobre los pesos de ResNet50 sigue presente al igual que en la arquitectura anterior.

El componente Decoder se aleja de la estructura recurrente y adopta la arquitectura Transformer. No es un único módulo monolítico, sino una composición de _varias capas especializadas_. La entrada textual (la secuencia de descripciones durante el entrenamiento o la generación) es procesada inicialmente por la Capa de Embedding Secuencial. Este módulo combina Embeddings de Tokens, que representan las palabras del vocabulario, con Embeddings Posicionales, que codifican la posición de cada palabra en la secuencia. Esta suma proporciona al modelo información sobre el orden de las palabras, una necesidad intrínseca dado que, a diferencia de las RNN, la auto-atención por sí misma no procesa secuencias de forma ordenada.

El núcleo del decodificador reside en una pila de Capas Decoder. Cada capa encapsula tres sub-mecanismos fundamentales. Primero, la Atención Auto-Causal (Causal Self-Attention), permite que cada posición en la secuencia de entrada textual atienda a las posiciones anteriores (incluida ella misma) dentro de la misma secuencia. Una máscara causal asegura que la predicción para una posición $t$ solo dependa de las salidas conocidas en posiciones anteriores ($<t$), preservando la naturaleza autorregresiva necesaria para la generación de texto. Segundo, la Atención Cruzada (Cross-Attention), es el mecanismo clave que conecta la modalidad visual con la textual. Aquí, las representaciones textuales generadas por la auto-atención (actuando como queries) atienden a las características de la imagen procesadas por el codificador (actuando como keys y values). Esto permite al decodificador enfocar selectivamente partes relevantes de la imagen al generar cada palabra de la descripción. Tercero, una Red Feed-Forward (FeedForward), consistente en dos capas lineales con una activación ReLU intermedia, procesa independientemente la salida de la atención cruzada en cada posición. 

Cada uno de estos tres sub-mecanismos dentro de la Capa Decoder incorpora conexiones residuales y normalización de capa para facilitar el flujo de gradientes y estabilizar el entrenamiento. Múltiples Capas Decoder se apilan para permitir que el modelo aprenda representaciones progresivamente más complejas.

La salida de esta última capa alimenta el módulo Output. Este consiste principalmente en una capa lineal que proyecta la representación final del tamaño de la capa oculta al tamaño del vocabulario, produciendo los logits para cada palabra posible. Esta capa tiene como característica distintiva el manejo explícito de Tokens Prohibidos (`PAD`, `SOS`, `UNK`), a cuyos índices se les asigna un sesgo fuertemente negativo ($-1"e9"$) para prevenir su generación durante la inferencia. Adicionalmente, se inicializa el sesgo de la capa lineal basándose en las frecuencias de las palabras en el vocabulario de entrenamiento con la idea de guiar el modelo en las etapas iniciales del entrenamiento.

El método forward del modelo completo orquesta el flujo de datos: codifica la imagen, reorganiza (rearrange) el mapa de características de la imagen de `[b, c, h, w]` a `[b, h*w, c]` para que sea compatible con la atención cruzada (tratando cada posición espacial como un "token" visual), aplica el embedding secuencial a las descripciones, procesa la secuencia a través de la pila de Capas Decoder (pasando tanto las características de imagen reorganizadas como los embeddings textuales a cada capa), y finalmente obtiene las logits del módulo Output.

La @resnet-attn ilustra la arquitectura del modelo y el flujo de datos a través de este. En dicho diagrama cada color de bloque indica lo siguiente:
- Blanco: Entrada
- Gris: Salida
- Verde: Capa del Encoder
- Azul: Capa de Embedding Secuencial
- Rojo: Capa del Decoder
- Amarillo: Capa de Salida (Output)

#figure(
  image("figures/Resnet-Attention.png"),
  caption: [Diagrama de la arquitectura y el flujo de datos en ResNet50-Attention.],
  placement: auto
) <resnet-attn>

En este diagrama se aclara que el tamaño de capa oculta es igual a la dimensionalidad de los embeddings de entrada (`embed_dim = hidden_size`), esto se debe a dos motivos. Primero, la consistencia dimensional facilita la conexión entre las diferentes partes del modelo. La salida de la capa de embedding y la salida proyectada del codificador ResNet tienen dimensión `embed_dim`. Las capas del decodificador operan internamente con una dimensión `hidden_size`. Si son iguales, estas representaciones pueden interactuar directamente (como en la Atención Cruzada) sin necesidad de capas lineales adicionales para ajustar las dimensiones.

Como segundo motivo se encuentra el hecho de tratarse de una práctica estándar: Es una convención establecida en muchas implementaciones de Transformers, incluyendo el artículo original "Attention Is All You Need"@attn. Se ha demostrado que funciona bien en la práctica y proporciona un buen equilibrio entre capacidad de representación y complejidad del modelo.

=== Encoder Swin

Tras haber explorado arquitecturas que se basan en la arquitectura ResNet50 como encoder, se consideró investigar el uso de un codificador de imágenes alternativo que representara un enfoque más reciente y basado fundamentalmente en mecanismos de atención. Con este objetivo, se seleccionó el Swin Transformer@swin preentrenado en ImageNet como el nuevo encoder a probar.

Similarmente a los enfoques anteriores, la arquitectura original es modificada descartándose la capa de clasificación original. Luego se aplica una capa de proyección implementada como una secuencia de convolución 1x1, seguida de una activación ReLU y Dropout espacial. La proyección transforma la dimensión de los canales de salida a la dimensión de embedding deseada. La capacidad de realizar fine-tuning sobre los pesos de Swin sigue presente al igual que en las arquitecturas anteriores.

Durante el paso hacia adelante (forward), la imagen de entrada es procesada primero por las capas de modificas de Swin. Luego, el tensor de características obtenido se pasa a través de la capa de proyección, resultando en un tensor de características de salida con la dimensión de embedding especificada.

Como era de interés en este proyecto, probar una arquitectura basada en su mayor parte en el mecanismo de atención, solo se utilizó como encoder al transformer Swin con el decoder basado en atención.

== Entrenamiento

El proceso de entrenamiento para el modelo sigue un enfoque estructurado que combina aprendizaje supervisado y, opcionalmente, aprendizaje por refuerzo. Como criterio central, se utiliza la pérdida de entropía cruzada (Cross Entropy Loss) para comparar las descripciones predichas con las reales durante el entrenamiento supervisado. El optimizador Adam se aplica cuando el decoder es basado en LSTM por su capacidad de adaptación, mientras que AdamW se usa cuando el decoder se encuentra basado en atención para mejorar la generalización. El planificador de tasa de aprendizaje (learning rate scheduler) disminuye la tasa si el rendimiento en validación se estanca.

En la etapa inicial, la fase de entrenamiento supervisada (con entropía cruzada), el modelo realiza pases hacia adelante y hacia atrás, aplicando recorte (clipping) de gradientes si es necesario para estabilizar el entrenamiento. La pérdida total de todos los tokens se suma y luego se normaliza en base al número de tokens procesados. El planificador ajusta la tasa de aprendizaje según el rendimiento en validación.

Si el modelo no mejora tras un número predefinido de épocas (según el parámetro de paciencia), y la configuración lo permite, el entrenamiento transiciona a la etapa de aprendizaje por refuerzo. En esta fase, el modelo optimiza la métrica CIDEr generando descripciones, calculando recompensas según su alineación con descripciones de referencia y utilizando métodos de policy gradient con técnicas de reducción de varianza.

== Generación 

La generación de descripciones para imágenes en los modelos presentados, tanto el basado en LSTM como el basado en la arquitectura Transformer, sigue un paradigma fundamentalmente autorregresivo. Una vez que el codificador ha procesado la imagen de entrada y ha extraído un conjunto de características representativas (features), el decodificador inicia la tarea de construir la descripción palabra por palabra. Este proceso comienza con un token especial de inicio de secuencia (`SOS`). A partir de este punto, en cada paso temporal, el decodificador utiliza las características de la imagen y la secuencia de palabras generadas hasta ese momento para predecir la probabilidad de la siguiente palabra en la secuencia.

Los modelos implementan dos estrategias principales para seleccionar la siguiente palabra: el muestreo basado en temperatura (temperature sampling) y la búsqueda por haz (beam search@beam). En el muestreo por temperatura, la distribución de probabilidad sobre el vocabulario, obtenida de la salida del decodificador (logits), se ajusta mediante un parámetro de temperatura. Una temperatura baja (cercana a cero) aproxima una selección determinista o greedy, eligiendo la palabra con la máxima probabilidad en cada paso. A medida que la temperatura aumenta, la distribución se suaviza, permitiendo un muestreo más estocástico y diverso, donde palabras menos probables tienen una mayor oportunidad de ser seleccionadas. La generación continúa hasta que se predice un token especial de fin de secuencia (`EOS`) o se alcanza una longitud máxima predefinida.

Alternativamente, beam search explora múltiples secuencias candidatas simultáneamente. En lugar de seleccionar solo la palabra más probable en cada paso, mantiene un número fijo (el tamaño del haz o "beam size") de las secuencias parciales más probables hasta ese momento. En el siguiente paso, cada una de estas secuencias se expande considerando las siguientes palabras más probables, y nuevamente se seleccionan las mejores secuencias acumuladas según su probabilidad total, normalizada por la longitud para evitar una preferencia excesiva por secuencias cortas. Este proceso se repite hasta que todas las secuencias del beam alcanzan el token `EOS` o la longitud máxima. Finalmente, se selecciona la secuencia completa con la mayor probabilidad normalizada como la descripción final.

== Herramientas y Hardware 

Con respecto al software, el lenguaje de programación principal fue Python 3.13. La implementación de las arquitecturas, así como los ciclos de entrenamiento y la inferencia, se llevaron a cabo sobre el framework PyTorch.

Para la gestión y preprocesamiento de los datos textuales asociados a las imágenes, se recurrió a bibliotecas especializadas en Procesamiento del Lenguaje Natural. Se utilizó NLTK (Natural Language Toolkit) para tareas estándar de manipulación de texto y SentencePiece para la tokenización a nivel Byte-Pair Encoding.

La monitorización de los experimentos, incluyendo el seguimiento en tiempo real de las métricas de pérdida y evaluación, así como la visualización de hiperparámetros y resultados, se gestionó a través de la plataforma Weights & Biases (WandB). Para la generación de gráficos estáticos destinados al análisis exploratorio de datos y la presentación de resultados, se emplearon las bibliotecas de visualización Matplotlib, Seaborn y Plotly.

Desde la perspectiva del hardware, se contó con GPU NVIDIA GeForce GTX 3070 y CPU Intel Core i7 de 7ª generación.
#pagebreak()

= Experimentos y Resultados <exp>

En total se realizaron 93 experimentos con los modelos ResNet50-LSTM, ResNet50-Attention y FullAttention (Swin+Transformer) sobre los datasets Flickr8k y COCO. Su distribución específica se detalla en la @experiments. 

En las siguientes sub-secciones se describen en particular los experimentos realizados y se analizan los resultados.

#figure(
  table(
    columns: (auto, 1fr, 1fr, 2fr),
    [Modelo],[Flickr8k],[COCO],[Total por Modelo],
    [ResNet50 + LSTM],[36],[2],[38],
    [ResNet50 + Transformer],[43],[10],[53],
    [Swin + Transformer],[1],[1],[2],
    table.hline(stroke: .5pt),
    [Total por Dataset],[80],[13],[93]
  ), 
  caption: [Distribución de los experimentos por modelo y por dataset.],
  placement: bottom
) <experiments>

== Dataset Flickr8k

Con el dataset Flickr8k se realizaron dos búsquedas bayesianas de hiperparámetros, una para el modelo ResNet50-LSTM y otra para el modelo ResNet50-Attention, con el objetivo de minimizar la pérdida en el conjunto de validación. La @sweep-params describe los valores que se probaron para cada parámetro, aclarando en que modelo fueron aplicados.

En estos barridos se utilizó la siguiente configuración:
- _Cantidad de épocas:_ 100 con una paciencia de 10.
- _Tamaño de beam:_ 5
- _Temperatura:_ 0
- _Largo máximo de descripción:_ 50
- _Fine-tune del encoder:_ Nulo
- _Scheduler:_ Nulo

#figure(
  table(
    columns: (auto, auto, auto),
    align: left,
    [Parámetro], [Valores], [Modelo Aplicado],
    [_Número de capas_], [1, 2, 3], [Ambos],
    [_Tamaño capa oculta_], [256, 512, 1024], [Ambos],
    [_Tamaño capa embedding_], [256, 512, 1024], [ResNet50-LSTM],
    [_Learning rate encoder_], [1e-05, 5e-05, 0.0001, 0.0005], [Ambos],
    [_Learning rate decoder_], [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005], [Ambos],
    [_Dropout encoder_], [0.1, 0.3, 0.5], [Ambos],
    [_Dropout decoder_], [0.1, 0.3, 0.5], [Ambos],
    [_Cabezas de atención_], [2, 4, 8], [ResNet50-Attention]
  ), 
  caption: [Valores probados en la búsqueda de hiperparámetros para los modelos ResNet50-LSTM y ResNet50-Attention sobre el dataset Flickr8k.],
  placement: bottom,
) <sweep-params>

Para analizar el desempeño de cada arquitectura, en la @loss-flickr se ilustran las curvas de evolución de la pérdida sobre el conjunto de validación. Estas permiten evaluar la estabilidad y velocidad de convergencia de los modelos ResNet50-LSTM y ResNet50-Attention frente a las diferentes configuraciones exploradas. Complementariamente, la @box-plot-flickr presenta diagramas de caja y bigotes para las métricas más representativas: la pérdida en validación, y los valores de CIDEr y BLEU-4 obtenidos en el conjunto de test.


#figure(
  grid(
    columns: 1,
    row-gutter: 3pt,
    image("figures/val_loss_ResNet50-LSTM_flickr8k.png", width: 100%),
    sub_caption[(a)],
    image("figures/val_loss_ResNet50-Attention_flickr8k.png", width: 100%),
    sub_caption[(b)],
  ),
  caption: [Pérdida en el conjunto de validación del dataset Flickr8k de los modelos (a) ResNet50-LSTM y (b) ResNet50-Attention.],
  gap: 12pt,
  placement: auto
) <loss-flickr>


#figure(
  grid(
    columns: 2,
    row-gutter: 3pt,
    image("figures/boxplot_val_loss.min_flickr8k.png", width: 8.5cm),
    image("figures/boxplot_test_BLEU-4.max_flickr8k.png", width: 8.5cm),
    sub_caption[(a)],
    sub_caption[(b)],
    grid.cell(colspan: 2, image("figures/boxplot_test_CIDEr.max_flickr8k.png", width: 8.5cm)),
    grid.cell(colspan: 2, sub_caption[(c)])
  ),
  caption: [Diagramas de caja y bigotes de las variables (a) pérdida en el conjunto de validación, (b) BLEU-4 y (c) CIDEr en el conjunto de test comparando los modelos ResNet50-LSTM y ResNet50-Attention sobre el dataset Flickr8k.],
  gap: 12pt,
  placement: auto
) <box-plot-flickr>

En los gráficos anteriores se observa que, en promedio, el modelo ResNet50-Attention obtiene valores ligeramente superiores en las métricas de generación CIDEr y BLEU-4 en comparación con ResNet50-LSTM. No obstante, este mejor desempeño viene acompañado de una mayor inestabilidad, evidenciada por una pérdida más dispersa en el conjunto de validación y una mayor presencia de valores atípicos en ambas métricas de evaluación. Esto sugiere que, si bien la atención puede potenciar la calidad de las descripciones generadas, también introduce variabilidad en los resultados, posiblemente debido a una mayor sensibilidad a la configuración de hiperparámetros.

Con el fin de analizar el efecto de los hiperparámetros sobre el rendimiento de los modelos, se generaron gráficos de coordenadas paralelas para cada métrica objetivo (@para-lstm[Figuras]@para-attn[ y]), lo que permite observar patrones en las combinaciones que conducen a mejores resultados. Complementariamente, se calcularon medidas de correlación e importancia por permutación para estimar la influencia individual de cada parámetro sobre las métricas de interés (@coor-lstm[Figuras]@coor-attn[ y]).

#figure(
  grid(
    columns: 1,
    row-gutter: 2pt,
    box(image("figures/val-loss_lstm_resnet_2.png"), clip: true, inset: (bottom: -0.5cm)),
    sub_caption[(a) ],
    box(image("figures/bleu4_lstm_resnet_plotly.png"), clip: true, inset: (bottom: -0.5cm)),
    sub_caption[(b)],
    box(image("figures/cider_lstm_resnet_plotly.png"), clip: true, inset: (bottom: -0.5cm)),
    sub_caption[(c)],
  ),
  caption: [Coordenadas paralelas del modelo ResNet50-LSTM sobre el dataset Flickr8k con variables objetivo (a) pérdida en el conjunto de validación, (b) BLEU-4 y (c) CIDEr en el conjunto de test.],
  gap: 12pt,
  placement: auto
) <para-lstm>

#figure(
  grid(
    columns: 1,
    row-gutter: 2pt,
    image("figures/ResNet50-LSTM_val_loss.min_correlation_importance.png"),
    sub_caption[(a)],
    image("figures/ResNet50-LSTM_test_BLEU-4.max_correlation_importance.png"),
    sub_caption[(b)],
    image("figures/ResNet50-LSTM_test_CIDEr.max_correlation_importance.png"),
    sub_caption[(c)],
  ),
  caption: [Correlación e importancia por permutación de parámetros en el modelo ResNet50-LSTM sobre el dataset Flickr8k con variables objetivo (a) pérdida en el conjunto de validación,(b) BLEU-4 y (c) CIDEr en el conjunto de test.],
  gap: 12pt,
  placement: auto,
) <coor-lstm>

En el modelo ResNet50-LSTM, la importancia por permutación sitúa a la tasa de aprendizaje en el decoder (Decoder LR) y al número de capas (Num Layers) como los hiperparámetros más influyentes en la pérdida de validación. Se observa que una mayor cantidad de capas o una tasa de aprendizaje más alta tienden a incrementar la pérdida de validación, lo que sugiere problemas de sobreajuste o convergencia. El dropout del encoder y un mayor tamaño de la capa oculta se correlacionan negativamente con la pérdida, indicando un efecto positivo en la regularización y la capacidad del modelo.

Con respecto a la métrica CIDEr, un mayor dropout en el decoder muestra una correlación positiva notable, sugiriendo una mejora en la calidad de las descripciones generadas. Por otro lado, tasas de aprendizaje elevadas en el encoder y tamaños de embedding pequeños se correlacionan negativamente con CIDEr. La importancia por permutación destaca nuevamente a la tasa de aprendizaje en el decoder como el parámetro más importante, seguido por el tamaño del embedding (Embed Size) y el dropout del decoder.

Al igual que con CIDEr, un mayor dropout en el decoder se correlaciona positivamente con BLEU-4, lo que indica una mejora en la fluidez de las descripciones generadas. La tasa de aprendizaje en el encoder y el tamaño del embedding presentan correlaciones negativas con esta métrica. La importancia por permutación subraya la relevancia de la tasa de aprendizaje en el decoder y el tamaño del embedding.

#figure(
  grid(
    columns: 1,
    row-gutter: 2pt,
    box(image("figures/val-loss_attn_resnet.png"), clip: true, inset: (bottom: -0.5cm)),
    sub_caption[(a)],
    box(image("figures/bleu4_attn_resnet.png"), clip: true, inset: (bottom: -0.5cm)),
    sub_caption[(b)],
    box(image("figures/cider_attn_resnet.png"), clip: true, inset: (bottom: -0.5cm)),
    sub_caption[(c)],
  ),
  caption: [Coordenadas paralelas del modelo ResNet50-Attention sobre el dataset Flickr8k con variables objetivo (a) pérdida en el conjunto de validación, (b) BLEU-4 y (c) CIDEr en el conjunto de test.],
  gap: 12pt,
  placement: auto,
) <para-attn>

#figure(
  grid(
    columns: 1,
    row-gutter: 2pt,
    image("figures/ResNet50-Attention_val_loss.min_correlation_importance.png"),
    sub_caption[(a)],
    image("figures/ResNet50-Attention_test_BLEU-4.max_correlation_importance.png"),
    sub_caption[(b)],
    image("figures/ResNet50-Attention_test_CIDEr.max_correlation_importance.png"),
    sub_caption[(c)],
  ),
  caption: [Correlación e importancia por permutación de parámetros en el modelo ResNet50-Attention sobre el dataset Flickr8k con variables objetivo (a) pérdida en el conjunto de validación, (b) BLEU-4 y (c) CIDEr en el conjunto de test.],
  gap: 12pt,
  placement: auto,
) <coor-attn>

Los resultados para el modelo ResNet50-Attention revelan nuevamente a la tasa de aprendizaje del decoder como el hiperparámetro dominante en todas las métricas analizadas.

Una alta tasa de aprendizaje del decoder se correlaciona positivamente con una mayor pérdida de validación, sugiriendo inestabilidad o sobreajuste durante el entrenamiento. La tasa de aprendizaje del encoder también muestra una correlación positiva moderada con la pérdida. En contraste, un mayor dropout en el encoder y, en menor medida, arquitecturas más profundas o con más cabezas, tienden a reducir la pérdida de validación. La importancia por permutación confirma que la tasa de aprendizaje del decoder es el parámetro más crítico para la estabilidad del entrenamiento.

Entre la tasa de aprendizaje del decoder y la métrica CIDEr existe una fuerte correlación negativa, indicando que tasas de aprendizaje elevadas en el decoder deterioran la calidad de las descripciones generadas. La tasa de aprendizaje del encoder también presenta una correlación negativa, aunque de menor magnitud. Un ligero aumento en el dropout tanto en el encoder como en el decoder parece favorecer la generalización y, por ende, mejorar el CIDEr. La importancia por permutación ratifica la primacía de la tasa de aprendizaje del decoder, seguido por el tamaño de la capa oculta (Hidden Size).

Los patrones observados en BLEU-4 son similares a los de CIDEr, con la tasa de aprendizaje del decoder y el enconder mostrando correlaciones negativas significativas. Un mayor dropout en el decoder exhibe una pequeña mejora en la fluidez y precisión de las descripciones, mientras que el impacto del dropout en el encoder es menor. La importancia por permutación vuelve a señalar a la tasa de aprendizaje del decoder como el factor más influyente.

A partir de este análisis, se seleccionaron las configuraciones para los modelos entrenados sobre el dataset Flickr8k, con las cuales se obtuvieron los mejores resultados. Las características de dichas arquitecturas, su entrenamiento y resultados se detallan en la @flickr-results.

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    [], [ResNet50-LSTM], [ResNet50-Attention],
    table.cell(colspan: 3)[*Arquitectura*],
    table.hline(stroke: .1pt),
    [Número de capas],[2],[2],
    [Tamaño capa oculta],[256],[512],
    [Tamaño capa de embeddings],[1024],[--],
    [Dropout en el encoder],[0.5],[0.1],
    [Dropout en el decoder],[0.5],[0.5],
    [Fine-tuning del encoder],[nulo],[parcial],
    [Número de cabezas de atención],[--],[2],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Entrenamiento*],
    table.hline(stroke: .1pt),
    [Criterio],[Cross Entropy Loss],[Cross Entropy Loss],
    [Optimizador],[Adam],[AdamW],
    [Learning rate encoder],[0.00001],[0.00001],
    [Learning rate decoder],[0.0001],[0.0001],
    [Clipping de gradientes],[2],[2],
    [Número de épocas],[44],[10],
    [Frecuencia mínima de palabra en vocabulario],[3],[3],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Métricas Obtenidas*],
    table.hline(stroke: .1pt),
    [Pérdida Mínima],[2.60],[2.62],
    [CIDEr],[0.37],[0.44],
    [BLEU-4],[0.15],[0.18],
    [BLEU-2],[0.35],[0.40],
    [BLEU-1],[0.53],[0.58],
  ),
  caption: [Arquitectura, entrenamiento y resultados obtenidos de los mejores modelos entrenados con el dataset Flickr8k.]
) <flickr-results>

A modo ilustrativo, en la @flickr-captions se presentan ejemplos de descripciones generadas por estos modelos. Las descripciones fueron obtenidas utilizando beam search con tamaño 5, y se aplicaron sobre un conjunto de imágenes nunca vistas por el modelo. Las primeras tres imágenes son de la autora, mientras el resto proviene de fuentes externas citadas a continuación de izquierda a derecha, arriba hacia abajo: (a) @img_farmers, (b) @img_kevin y (c) @img_kingfisher.

#figure(
  grid(
    columns: 2,
    row-gutter: 3pt,
    image("pics/flickr8k_001.png", width:7.5cm),
    image("pics/flickr8k_002.png", width:7.5cm),
    image("pics/flickr8k_003.png", width:7.5cm),
    // image("pics/flickr8k_004.png", width:7.5cm),
    image("pics/flickr8k_005.png", width:7.5cm),
    image("pics/flickr8k_006.png", width:7.5cm),
    image("pics/flickr8k_007.png", width:7.5cm),
  ),
  caption: [Descripciones generadas con los modelos ResNet50-LSTM y ResNet50-Attention entrenados en el dataset Flickr8k. ],
  placement: auto,
) <flickr-captions>

Si bien la estructura de las oraciones es coherente y algunas palabras clave se corresponden con el contenido visual, puede observarse que la calidad general de las descripciones no es del todo satisfactoria. En varios casos, las frases resultan genéricas, contienen errores semánticos o simplemente describen erróneamente la imagen, lo cual evidencia las limitaciones del modelo cuando se entrena sobre un dataset de tamaño reducido como Flickr8k.

== Dataset COCO

Dado que los resultados obtenidos con el dataset Flickr8k no fueron del todo satisfactorios, se decidió evaluar ambos modelos sobre un conjunto de datos más extenso y representativo: COCO. La configuración de hiperparámetros aplicada en este caso se definió a partir del análisis de los resultados previos con Flickr8k. No se realizó una búsqueda bayesiana de hiperparámetros para COCO, ya que su gran tamaño implica un costo computacional considerable —más de una hora por época de entrenamiento—, lo cual excedía los recursos disponibles.

Al igual que en el experimento anterior, se analiza el comportamiento de los modelos mediante la evolución de la pérdida en el conjunto de validación, como se muestra en la @loss-coco. Asimismo, la @boxplot-coco presenta diagramas de caja y bigotes para las métricas de mayor relevancia: pérdida en validación, CIDEr y BLEU-4 en el conjunto de test. Cabe aclarar que estas visualizaciones corresponden exclusivamente al modelo ResNet50-Attention.

En cuanto al modelo ResNet50-LSTM, solo se obtuvieron dos resultados, por lo que se presentan de forma tabular en la @results-coco-lstm. Aunque su curva de pérdida es única, se consideraron dos checkpoints distintos: el correspondiente a la mínima pérdida alcanzada y el último al finalizar el entrenamiento, para así comparar sus métricas de evaluación.

#figure(
  image("figures/val_loss_ResNet50-LSTM_ResNet50-Attention_coco.png"),
  caption: [Pérdida en el conjunto de validación del dataset COCO de los modelos ResNet50-LSTM y ResNet50-Attention.],
  gap: 12pt,
  // placement: auto
) <loss-coco>

#figure(
  grid(
    columns: 3,
    row-gutter: 2pt,
    image("figures/boxplot_val_loss.min_coco.png"),
    image("figures/boxplot_test_CIDEr.max_coco.png"),
    image("figures/boxplot_test_BLEU-4.max_coco.png"),
    sub_caption[(a)],
    sub_caption[(b)],
    sub_caption[(c)],
  ),
  caption: [Diagramas de caja y bigotes de las variables (a) pérdida en el conjunto de validación, (b) CIDEr y (c) BLEU-4 en el conjunto de test con el modelo ResNet50-Attention sobre el dataset COCO.],
  gap: 12pt,
  // placement: top,
) <boxplot-coco>

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr),
    align: left,
    [Métrica], [Mínimo], [Máximo], [Promedio],
    [_Pérdida en validación_], [2.35025], [2.35025], [2.35025],
    [_CIDEr en test_], [0.68463], [0.69743], [0.69103], 
    [_BLEU-4 en test_], [0.23958], [0.2446], [0.24209]
  ), 
  caption: [Resultados obtenidos para el modelo ResNet50-LSTM sobre el dataset COCO.],
  // placement: top,
) <results-coco-lstm>

Por otra parte, tomando los modelos que obtuvieron los mayores valores en CIDEr y BLEU-4, se generaron nuevamente descripciones para las imágenes presentadas anteriormente en la @flickr-captions, esta vez utilizando los modelos entrenados sobre COCO. Las descripciones generadas pueden observarse en la @coco-captions, y los detalles de los modelos: sus respectivas arquitecturas, configuraciones de entrenamiento y resultados alcanzados se resumen en la @coco-results.

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    [], [ResNet50-LSTM], [ResNet50-Attention],
    table.cell(colspan: 3)[*Arquitectura*],
    table.hline(stroke: .1pt),
    [Número de capas],[3],[2],
    [Tamaño capa oculta],[512],[512],
    [Tamaño capa de embeddings],[512],[--],
    [Dropout en el encoder],[0.1],[0.2],
    [Dropout en el decoder],[0.5],[0.5],
    [Fine-tuning del encoder],[parcial],[parcial],
    [Número de cabezas de atención],[--],[4],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Entrenamiento*],
    table.hline(stroke: .1pt),
    [Criterio],[Cross Entropy Loss],[Cross Entropy Loss],
    [Optimizador],[Adam],[AdamW],
    [Learning rate encoder],[0.00001],[0.00001],
    [Learning rate decoder],[0.0001],[0.0001],
    [Clipping de gradientes],[2],[1],
    [Paciencia],[10 épocas],[10 épocas],
    [Número de épocas],[10],[6],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Vocabulario*],
    table.hline(stroke: .1pt),
    [Estrategia],[BPE],[Palabra],
    [Tamaño], [8500 piezas], [Frecuencia mín. de palabra: 5],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Métricas Obtenidas*],
    table.hline(stroke: .1pt),
    [Pérdida Mínima],[2.35],[2.17],
    [CIDEr],[0.70],[0.73],
    [BLEU-4],[0.24],[0.26],
    [BLEU-2],[0.47],[0.48],
    [BLEU-1],[0.64],[0.65],
  ),
  caption: [Arquitectura, entrenamiento y resultados obtenidos de los mejores modelos entrenados con el dataset COCO.]
) <coco-results>

#figure(
  grid(
    columns: 2,
    row-gutter: 3pt,
    image("pics/coco_001.png", width:7.5cm),
    image("pics/coco_002.png", width:7.5cm),
    image("pics/coco_003.png", width:7.5cm),
    // image("pics/coco_004.png", width:7.5cm),
    image("pics/coco_005.png", width:7.5cm),
    image("pics/coco_006.png", width:7.5cm),
    image("pics/coco_007.png", width:7.5cm),
  ),
  caption: [Descripciones generadas con los modelos ResNet50-LSTM y ResNet50-Attention entrenados en el dataset COCO.],
  placement: auto,
) <coco-captions>

Aunque las descripciones generadas por los modelos entrenados con COCO no son perfectas, se observa una mejora significativa respecto a las obtenidas con Flickr8k. Las frases son más específicas, mejor estructuradas y tienden a reflejar de manera más precisa el contenido visual.

En la @flickr-vs-coco se muestra una comparación directa de las métricas obtenidas por los mejores modelos de cada dataset. Se evidencia una clara evolución en el desempeño, particularmente en las métricas CIDEr y BLEU-4, al entrenar sobre un corpus más extenso como COCO.

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr),
    align: left,
    [Dataset], [Pérdida Mínima], [CIDEr Máximo], [BLEU-4 Máximo],
    [_Flickr8k_], [2.62], [0.44], [0.18],
    [_COCO_], [2.17], [0.73], [0.26],
  ), 
  caption: [Comparación de las métricas obtenidas en los datasets Flickr8k y COCO.],
  placement: auto,
) <flickr-vs-coco>

== Encoder Swin y Modelo FullAttention

En el espíritu de explorar nuevas arquitecturas, se realizaron dos nuevos experimentos utilizando el Swin Transformer como encoder. Estas pruebas se llevaron a cabo tanto en los datasets Flickr8k como COCO, manteniendo como decoder el módulo de atención, con el fin de evaluar el rendimiento de un modelo completamente basado principalmente en mecanismos de atención (FullAttention).

A continuación, en la @swin-loss se muestra la evolución de la pérdida en el conjunto de validación para ambos datasets.

#figure(
  image("figures/val_loss_Swin-Attention_flickr8k_coco.png", width: 14cm),
  caption: [Pérdida del modelo FullAttention en el conjunto de validación de los datasets Flickr8k y COCO.]
) <swin-loss>

En la tabla @swin-results se resumen las características del modelo, los parámetros de entrenamiento y los resultados obtenidos en ambos datasets:

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    [], [Flickr8k], [COCO],
    table.cell(colspan: 3)[*Arquitectura*],
    table.hline(stroke: .1pt),
    [Número de capas],table.cell(colspan: 2)[2],
    [Tamaño capa oculta],table.cell(colspan: 2)[512],
    [Dropout en el encoder],table.cell(colspan: 2)[0.2],
    [Dropout en el decoder],table.cell(colspan: 2)[0.5],
    [Fine-tuning del encoder],table.cell(colspan: 2)[parcial],
    [Número de cabezas de atención],table.cell(colspan: 2)[4],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Entrenamiento*],
    table.hline(stroke: .1pt),
    [Criterio],table.cell(colspan: 2)[Cross Entropy Loss],
    [Optimizador],table.cell(colspan: 2)[AdamW],
    [Learning rate encoder],table.cell(colspan: 2)[0.00001],
    [Learning rate decoder],table.cell(colspan: 2)[0.0001],
    [Clipping de gradientes],table.cell(colspan: 2)[1],
    [Paciencia],table.cell(colspan: 2)[10 épocas],
    [Número de épocas],[21],[19],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Vocabulario*],
    table.hline(stroke: .1pt),
    [Estrategia],table.cell(colspan: 2)[Palabra],
    [Tamaño], table.cell(colspan: 2)[Frecuencia mínima de palabra: 5],
    table.hline(stroke: .1pt),
    table.cell(colspan: 3)[*Métricas Obtenidas*],
    table.hline(stroke: .1pt),
    [Pérdida Mínima],[2.42],[2.17],
    [CIDEr],[0.50],[0.75],
    [BLEU-4],[0.20],[0.26],
    [BLEU-2],[0.42],[0.49],
    [BLEU-1],[0.60],[0.66],
  ),
  caption: [Arquitectura, entrenamiento y resultados obtenidos del modelo FullAttention sobre el dataset Flickr8k y COCO.],
  placement: auto
) <swin-results>

Si bien los resultados cuantitativos presentados en la @swin-results muestran un incremento modesto en las métricas respecto a otros modelos, se observó una mejora cualitativa en las descripciones generadas, especialmente en la capacidad del modelo para identificar elementos relevantes en las imágenes, lo cual fue más evidente en el dataset Flickr8k. Finalmente, se presentan en la @swin-captions algunas de las descripciones generadas por el modelo FullAttention.

#figure(
  grid(
    columns: 2,
    row-gutter: 3pt,
    image("pics/swin_001.png", width:7.5cm),
    image("pics/swin_002.png", width:7.5cm),
    image("pics/swin_003.png", width:7.5cm),
    // image("pics/swin_004.png", width:7.5cm),
    image("pics/swin_005.png", width:7.5cm),
    image("pics/swin_006.png", width:7.5cm),
    image("pics/swin_007.png", width:7.5cm),
  ),
  caption: [Descripciones generadas con el modelo FullAttention sobre el dataset Flickr8k y sobre COCO.],
  placement: auto,
) <swin-captions>

== Interpretabilidad mediante Mapas de Atención

Con el objetivo de interpretar el comportamiento del mecanismo de atención, se generaron visualizaciones de los mapas de atención correspondientes a distintos momentos del proceso de descripción. Estas imágenes permiten observar qué regiones de la imagen son destacadas por el modelo al generar cada palabra. Las zonas más claras indican una mayor atención asignada en ese instante.

En la @t_attn[Figuras]@swin_attn[ y] se presentan ejemplos representativos obtenidos tanto del modelo FullAttention como del modelo ResNet50-Attention.

#figure(
  grid(
    columns: 1,
    image("pics/coco_t_attention_007.png"),
  ),
  caption: [Mapas de atención generados por el modelo RestNet50-Attention entrenado con el dataset COCO. Las zonas más claras indican mayor atención del modelo al momento de generar ciertas palabras.], 
  placement: auto,
) <t_attn>

#figure(
  grid(
    columns: 1,
    image("pics/swin_coco_attention_002.png")
  ),
  caption: [Mapas de atención generados por el modelo FullAttention entrenado con el dataset COCO. Las zonas más claras indican mayor atención del modelo al momento de generar ciertas palabras.], 
  placement: auto,
) <swin_attn>

En estas figuras, se observa que, si bien se puede visualizar el cambio en la atención por palabra generada, este cambio es pequeño y, en muchas ocasiones, no se enfoca en los elementos que deberían estar siendo descritos. Esta falta de alineación entre la atención visual y los objetos relevantes en la imagen se correlaciona con los resultados cuantitativos obtenidos, lo cual sugiere que aún hay margen de mejora en la integración entre visión y lenguaje.

#pagebreak()

= Conclusiones y Trabajo Futuro <conclusiones>

A lo largo de los 93 experimentos realizados con las arquitecturas ResNet50-LSTM, ResNet50-Attention y FullAttention, y evaluados sobre los datasets Flickr8k y COCO, se obtuvieron varias conclusiones  en cuanto a rendimiento, comportamiento y viabilidad de estos modelos para tareas de descripción de imágenes.

En primer lugar, el tamaño del corpus tuvo un impacto considerable en el desempeño de los modelos. Con Flickr8k, un dataset pequeño (\~8000 imágenes), los resultados fueron limitados: el mejor modelo ResNet50-Attention alcanzó un CIDEr máximo de 0.44 y un BLEU‑4 de 0.18, mientras que el mejor ResNet50‑LSTM logró 0.37 y 0.15, respectivamente. Al utilizar COCO, un corpus mucho más grande (\~120000 imágenes), ambos modelos mejoraron sustancialmente sus métricas: ResNet50‑Attention llegó a CIDEr 0.73 y BLEU‑4 0.26, y FullAttention superó ligeramente estos valores con CIDEr 0.75 y BLEU‑4 0.26.

Respecto a la comparación entre LSTM y atención, el modelo LSTM mostró curvas de pérdida más estables y menor varianza, lo que sugiere un entrenamiento más predecible, aunque con menor capacidad generativa. Por otro lado, los modelos basados en mecanismos de atención, como ResNet50‑Attention y FullAttention, ofrecieron descripciones medianamente más precisas y coherentes, en especial al detectar objetos y relaciones espaciales, aunque presentaron una mayor inestabilidad en el entrenamiento, evidenciada por la varianza en los resultados de validación y la presencia de anómalos.

El reemplazo de ResNet50 por Swin Transformer como encoder en el modelo FullAttention permitió una mejora cualitativa en la identificación de detalles finos, como fondos complejos. Sin embargo, este enfoque incrementó el costo computacional de manera notable, con entrenamientos más largos y un mayor consumo de memoria, lo cual limitó su aplicabilidad en este entorno con recursos restringidos.

Entre las principales limitaciones observadas se encuentran la sensibilidad de los modelos a los hiperparámetros. Asimismo, la necesidad de grandes volúmenes de datos para que los modelos basados en atención generalicen de forma adecuada quedó clara, dado que Flickr8k fue insuficiente para capturar la diversidad lingüística y visual deseada. Otro obstáculo fue el tiempo de entrenamiento: la imposibilidad de realizar búsquedas bayesianas exhaustivas en COCO restringió la optimización de hiperparámetros y dejó espacio para mejoras adicionales. 

Entre las líneas de investigación pendientes se encuentra la exploración y comparación de distintos tamaños del vocabulario para un mismo dataset, y de distintos tamaños de beam en la generación de descripciones. Adicionalmente, aunque se implementó un mecanismo para el entrenamiento mediante aprendizaje por refuerzo utilizando CIDEr como recompensa, las pruebas iniciales arrojaron resultados deficientes y consumieron un tiempo y memoria considerable. Por lo tanto, la exploración y ajuste de esta técnica quedó como una tarea pendiente relevante, dado que ha demostrado buenos resultados en otros trabajos de investigación sobre descripción de imágenes.@mesh

Para trabajos futuros, se podría explorar el preentrenamiento y la transferencia de modelos, aprovechando encoders y decoders preentrenados en tareas de visión-lenguaje lo cual podría reducir la necesidad de grandes corpus y acelerar la convergencia. También, se podría aplicar estrategias más eficientes de optimización de hiperparámetros, como búsquedas sucesivas o aprendizaje de hiperparámetros. 

En conclusión, los mecanismos de atención demostraron un potencial para elevar la calidad de las descripciones generadas, especialmente cuando se dispone de un corpus amplio como COCO, pero requieren un cuidadoso ajuste y recursos de cómputo considerables. El modelo FullAttention con Swin Transformer aporta mejoras adicionales, aunque de manera incremental frente a ResNet50‑Attention. Si bien se esperaba una mejora significativa con el uso de atención, especialmente en COCO, el incremento en las métricas no fue tan notable en proporción al tiempo de entrenamiento requerido. Esto sugiere que, en ciertos contextos o con recursos limitados, puede preferirse trabajar con datasets más pequeños como Flickr8k o con arquitecturas más simples, siempre que se adapten adecuadamente al dominio de aplicación.

#pagebreak()

#bibliography(
	"bib.yml",
	full: false
)

#pagebreak()
#appendix[
  = Anexo
  == Mapas de Atención
  #figure(
    image("pics/coco_t_attention_001.png"),
    caption: [Mapas de atención generados por el modelo RestNet50-Attention entrenado con el dataset COCO.]
  )
  #figure(
    image("pics/coco_t_attention_003.png"),
    caption: [Mapas de atención generados por el modelo RestNet50-Attention entrenado con el dataset COCO.]
  )
  #figure(
    image("pics/swin_coco_attention_003.png"),  
    caption: [Mapas de atención generados por el modelo FullAttention entrenado con el dataset COCO.]
  )
]


