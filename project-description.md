# Neural Echo

## ¿Qué es?

Un optimizador de música con feedback de cerebro artificial: subes una canción y una restricción creativa en lenguaje natural, y un agente de IA remezcla la pista hasta que un modelo de encoding cerebral confirma que estimula sensorialmente igual al contenido original.

## Descripción

NeuralEcho resuelve un problema de las herramientas de generación musical con IA: no hay forma objetiva de saber si una nueva composición realmente *se siente* como la referencia que el usuario tenía en mente.

El sistema corre un loop de optimización cerrado: 
→ Un agente propone un plan de composición en ElevenLabs Music v2 
→ El plan se renderiza a audio → el audio se puntúa comparando su respuesta cerebral predicha (usando el modelo de encoding cerebral TRIBE v2, de Meta) contra la de la canción de referencia 
→ Los puntajes y diagnósticos por red cerebral vuelven a Claude, que escribe una hipótesis y propone el siguiente plan 
→ El ciclo se repite hasta convergencia, un límite de generaciones o un límite de paciencia. El usuario ve la evolución en vivo y descarga la versión ganadora.

La interfaz es "upload-first": subes tu canción de referencia y tu restricción, y el sistema itera solo. El input por URL de YouTube está visible en la interfaz pero marcado como "próximamente" y no puede seleccionarse todavía.

## Arquitectura


neural_echo/       compat.py    shims de compatibilidad con TRIBE
                    ingest.py    YouTube/upload -> clip de referencia normalizado (90s)
                    metric.py    la función de costo cerebral (región x ventana temporal)
                    atlases.py   atlas Destrieux (fsaverage5) -> regiones anatómicas
                    generator.py schema del "genoma" + cliente async de ElevenLabs
                    analysis.py  features de librosa para el plan inicial de Claude
                    optimizer.py el loop de optimización (single-lineage, LangGraph)
services/api/       app FastAPI: envío de jobs, stream de progreso (SSE), artefactos
apps/web/           frontend en Next.js (3 pantallas: setup, evolución, resultado)
data/clip_library/  librería de clips de referencia (usada por los tests)
tests/               gate de validación de la métrica


## Stack

- *Claude Sonnet 5* — razona sobre los diagnósticos y propone cada plan de composición
- *ElevenLabs Music v2* — renderiza cada plan a audio
- *TRIBE v2 (Meta)* — modelo de encoding cerebral que predice la respuesta neuronal a cada clip
- *FastAPI + LangGraph* — orquestación del loop de optimización y la API
- *Next.js* — frontend con las tres pantallas del flujo

## Consideraciones

- La vista de superficie cortical acompaña el run en vivo: muestra el cerebro de referencia y el del candidato lado a lado, y anima la evolución del candidato a través de las iteraciones del optimizador.

## Track

🌐 Simulations

## Equipo

- Daniel Vargas ([@dcsand](https://github.com/dcsand))
- Juan David Ramírez Jimenez ([@judramirezj](https://github.com/judramirezj))
- Sebastian Cuellar Harker ([@sebascuha](https://github.com/sebascuha))