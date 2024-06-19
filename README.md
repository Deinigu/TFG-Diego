# Detection and classification of pieces in a real chess board ♟️
<p align="center"><img src="https://res.cloudinary.com/dx4oicqhy/image/upload/v1718800058/github_tfg_portada.png" width="65%" height="65%"/>

This project was created by Diego López Reduello under the tutoring of Miguel Ángel Molina Cabello and Karl Khader Thurnhofer Hemsi for the Final Degree Project of the Software Engineering Degree at the University of Málaga.

## Description 📋
This project aims to develop a system composed of a neural network capable of detecting and classifying the different chess pieces arranged on a chessboard, as well as their positions on the board itself. Using digital image processing techniques, the system identifies the position of each piece and provides this information in FEN (Forsyth-Edwards Notation), which allows to easily reproduce the layout of the board.

## Key Features ✨

- **Piece Detection:** The system can identify different chess pieces (pawns, knights, bishops, rooks, queens and kings) and their color (white or black). 
- Piece Classification:** Accurate classification of the detected pieces according to their type and color. 
- Board Recognition:** Detection and mapping of the chessboard grid. 
- **Output in FEN Notation:** Generation of a string in FEN notation that represents the exact arrangement of the pieces on the board, thus facilitating their analysis and reproduction.
## Starting 🚀
### Requirements 📋

Before starting the installation, make sure that your system meets the following requirements: 

- **Operating System**: Windows, Linux, macOS. 
- Python 3.8 or higher 
- `pip` (Python package manager) 
- Git` (optional, to clone the repository) 
### Installation ⚙️
#### Clone the Repository (Optional) 📂

If you want to clone the application repository from GitHub, run the following command: 
```bash
 git clone https://github.com/Deinigu/TFG-Diego.git
```
#### Create a Virtual Environment 🛠️

It is recommended to create a virtual environment to avoid conflicts with other dependencies. Use the following commands:

- To create the environment:

```bash
python -m venv environment-name
 ```

- To activate it on Linux/macOS:

```bash
source environment-name/bin/activate
```

- To activate on Windows:

```bash
.environment-name/bin/scripts/activate
```
#### Install dependencies 📦

Install the necessary dependencies for the application using `pip`:

```bash
pip install -r requirements.txt
```
### Run ▶️

To verify that the installation process was successful, run the following command on the repository path:

```bash
python main.py -h
```

Executing this command should return a list of the different console parameters that can be used for the application. If not, something has gone wrong during installation and you should check it.

## Authors 👥

This project was done by:
 * Diego Lopez Reduello 

Under the tutoring of: 
* Miguel Angel Molina Cabello 
* Karl Khader Thurnhofer Hemsi

## License 📄
This project is under the MIT License - See the file [LICENSE](LICENSE) for more details.
