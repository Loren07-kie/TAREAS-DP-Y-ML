# ÓRBITA · Laboratorio de Machine Learning y Deep Learning

Cuatro proyectos de las semanas 9 a 12, organizados por tarea. El reporte solicitado está en Jupyter y contiene fundamento, diagrama de flujo de la función principal, código, resultados ejecutados, interpretación, conclusiones y referencias.

| Tarea | Semana | Proyecto | Carpeta |
|---|---|---|---|
| 1 | 9 | Regresión lineal simple y múltiple | tareas/Tarea_1_Semana_09 |
| 2 | 10 | Segmentación con K-Means | tareas/Tarea_2_Semana_10 |
| 3 | 11 | Red neuronal TensorFlow/Keras | tareas/Tarea_3_Semana_11 |
| 4 | 12 | Detección y reconocimiento de figuras | tareas/Tarea_4_Semana_12 |

## Entregas
Los cuatro ZIP están en web/descargas. Cada uno incluye Reporte.ipynb, proyecto.py, requirements.txt, datos de entrada y resultados. Se recomienda Python 3.12. Ver el README de cada tarea para ejecutar en Jupyter o Google Colab.

## Página interactiva
web/index.html contiene un portafolio adaptable con letras gruesas, bordes y efectos 3D. Incluye cuatro experimentos: predicción con coeficientes entrenados, K-Means por iteraciones, inferencia con pesos reales exportados de Keras y segmentación por componentes conexos. Las demostraciones se ejecutan en el navegador. La de K-Means usa otra muestra sintética y la de visión incorpora distintas intensidades para explorar el umbral.

Para vista local: python -m http.server 8765 --directory web. En Render: Static Site, rama main, comando de construcción echo Sitio listo y directorio publicado web. No requiere base de datos ni un servidor de entrenamiento.

Todos los datos son sintéticos. Consultar los límites y las referencias oficiales de cada cuaderno. Los espacios de nombre, curso y docente se dejan para completar con información real.

## Versión guiada y entrenamiento ampliado
El inicio integra los modelos para eliminar la espera por descargas JSON y usa fuentes del sistema. El fondo de la red se calcula una vez y se reutiliza al mover el punto. El logo y los cuatro esquemas son imágenes SVG locales. Cada caso incluye entradas, proceso, salida, usos, ejemplos con un clic y un minirreto.

La red se reentrenó con 3000 puntos sintéticos y ruido 0.12: 1800 para entrenamiento, 600 para validación y 600 de prueba. Se compararon capas 16–8 y 24–12 por pérdida de validación y se eligió 24–12. Acertó 598/600 puntos de prueba (99.67 %). No se compara directamente con la exactitud previa porque cambió el conjunto y su dificultad. El cuaderno y el ZIP de la tarea 3 contienen esta nueva ejecución. La interacción consulta el modelo; no recopila datos ni entrena automáticamente.

Validación de las interacciones: node verificar_web.cjs. Incluye igualdad de inferencia Keras/JavaScript, selección de ejemplos, minirretos, fondo cacheado, convergencia K-Means y segmentación. La revisión visual automatizada no estuvo disponible por un error del entorno.
