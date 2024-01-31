# Plagiarism Detection Project

This project aims to detect plagiarism between two given texts using a pre-trained machine learning model.

## Getting Started

### Prerequisites

Make sure you have the following installed on your machine:

- Python (>=3.11.6)
- Node.js (>=20.10.0)
- Pip (Python package installer >=23.3.2)
- Git

### Installation

1. **Clone the repository:**

   ```git
   git clone https://github.com/abdelhak002/plagiarism-detector-cnn.git
   ```

2. **Change into the project directory:**

   ```bash
   cd plagiarism-detection-cnn
   ```

3. **Create a virtual environment (optional but recommended):**

   ```python
   python -m venv .venv
   ```

4. **Activate the virtual environment:**

   - **On Windows:**

   ```bash
   .\.venv\Scripts\activate
   ```

   - **On macOS/Linux:**

   ```bash
   source .venv/bin/activate
   ```

5. **Install project dependencies:**

   ```python
   pip install -r requirements.txt
   ```

   ```javascript
   npm install
   ```

6. **Training the model:**

   ```python
   python model/build_model.py
   ```

7. **Testing the model:**

   ```python
   python test_model.py
   ```

## Flask Web App:

we load the pre-trained model and tokenizer from the `model` directory and use them in the flask web app.

to run the flask web app:

```python
flask --app app run
/or
python -m flask --app app run
```

debug mode:

```python
flask --app app run --debug
/or
python -m flask --app app run --debug
```
