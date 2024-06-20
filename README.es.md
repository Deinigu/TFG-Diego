# Detección y clasificación de piezas en un tablero real de ajedrez ♟️
<p align="center"><img src="https://res.cloudinary.com/dx4oicqhy/image/upload/v1718800058/github_tfg_portada.png" width="65%" height="65%"/> 

Este proyecto fue creado por Diego López Reduello bajo la tutorización de Miguel Ángel Molina Cabello y Karl Khader Thurnhofer Hemsi para la realización del Trabajo de Fin de Grado del Grado en Ingeniería del Software en la Universidad de Málaga.</p>

## Descripción 📋
Este proyecto tiene como finalidad desarrollar un sistema compuesto de una red neuronal capaz de detectar y clasificar las diferentes piezas de ajedrez dispuestas sobre un tablero, así como sus posiciones en el propio tablero. Utilizando técnicas de procesamiento digital de imágenes, el sistema identifica la posición de cada pieza y proporciona esa información en notación FEN (Forsyth-Edwards Notation), lo que permite reproducir fácilmente la disposición del tablero.

## Funcionalidades Clave ✨

- **Detección de Piezas:** El sistema puede identificar diferentes piezas de ajedrez (peones, caballos, alfiles, torres, reinas y reyes) y su color (blanco o negro). 
- **Clasificación de Piezas:** Clasificación precisa de las piezas detectadas en función de su tipo y color. 
- **Reconocimiento del Tablero:** Detección y mapeo de la cuadrícula del tablero de ajedrez. 
- **Salida en Notación FEN:** Generación de una cadena en notación FEN que representa la disposición exacta de las piezas en el tablero, facilitando así su análisis y reproducción.
## Comenzando 🚀
### Requisitos 📋

Antes de comenzar con la instalación, asegúrese de que su sistema cumpla con los siguientes requisitos: 

- **Sistema operativo**: Windows, Linux, macOS 
- Python 3.8 o superior 
- `pip` (gestor de paquetes de Python) 
- `Git` (opcional, para clonar el repositorio) 
### Instalación ⚙️
#### Clonar el Repositorio (Opcional) 📂

Si desea clonar el repositorio de la aplicación desde GitHub, ejecute el siguiente comando: 
```bash
 git clone https://github.com/Deinigu/TFG-Diego.git
 ```
#### Crear un Entorno Virtual 🛠️

Es recomendable crear un entorno virtual para evitar conflictos con otras dependencias. Use los siguientes comandos:

- Para crear el entorno:

```bash
python -m venv nombre-del-entorno
 ```

- Para activarlo en Linux/macOS:

```bash
source nombre-del-entorno/bin/activate
 ```

- Para activarlo en Windows:

```bash
.\nombre-del-entorno\Scripts\activate
```
#### Instalar Dependencias 📦

Instale las dependencias necesarias para la aplicación usando `pip`:

```bash
pip install -r requirements.txt
```
### Ejecución ▶️

Para comprobar que el proceso de instalación ha sido correcto, ejecute el siguiente comando en la ruta del repositorio:

```bash
python main.py -h
```

La ejecución de este comando debería devolver una lista con los diferentes parámetros de consola que se pueden utilizar para la aplicación. En caso contrario, algo ha salido mal durante la instalación y debería revisarlo.

## Autores 👥

Este proyecto fue realizado por:
 * Diego López Reduello 

Bajo la tutorización de: 
* Miguel Ángel Molina Cabello 
* Karl Khader Thurnhofer Hemsi
