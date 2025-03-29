
# Vedic-Pedia-AI

## Overview
Vedic-Pedia-AI is an AI-powered system designed to process and analyze documents efficiently. It provides functionalities to upload documents and query the system for relevant information.

## How to Run

### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/Vedic-Pedia-AI.git
cd Vedic-Pedia-AI
```

### 2. Create a `.env` File
Set up the required environment variables by creating a `.env` file in the root directory.

### 3. Configure API Keys
Add all necessary API keys and configuration settings inside the `.env` file.

### 4. Start the Server
Run the following command to start the API server using Uvicorn:
```sh
uvicorn api:app --reload
```

## API Routes

### 1. Upload Documents
**Endpoint:**
```http
POST /upload
```
**Description:** Uploads a document to the system.

### 2. Query the System
**Endpoint:**
```http
GET /query?text=your_query
```
**Description:** Queries the system for relevant information based on the provided input.

---

## Requirements
Ensure you have the following dependencies installed:
```sh
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.

---

### Contributors
- Your Name (@your-handle)

---

For any issues, feel free to open a ticket or reach out!