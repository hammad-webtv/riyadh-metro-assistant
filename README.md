# Riyadh Metro Assistant

A comprehensive AI-powered assistant for the Riyadh Metro system, providing information about metro routes, stations, amenities, sports venues, and more.

## Features

- **Metro Navigation**: Get directions and information about Riyadh Metro lines and stations
- **Station Amenities**: Find information about facilities, restaurants, and services near metro stations
- **Sports Venues**: Get details about football stadiums and their metro connections
- **Metro Rules & Etiquette**: Learn about train classes, safety rules, and cultural guidelines
- **Multi-language Support**: Responds in Arabic and English
- **Contextual Search**: Intelligent search across all data types

## Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Git (for cloning the repository)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pharmacy-assistant-main
   ```

2. **Install dependencies**:
   
   **Option A: Using pip**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Option B: Using uv (recommended)**
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   
   Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
   
   Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Data Indexing

Before running the application, you need to index all the data files into the vector database:

1. **Start the application**:
   ```bash
   cd repository
   python main.py
   ```

2. **Index all data** (in a new terminal or via API call):
   
   **Option A: Using curl**
   ```bash
   curl -X POST http://localhost:3000/index_all_data
   ```
   
   **Option B: Using the web interface**
   - Open your browser and go to `http://localhost:3000`
   - Use the API endpoint `/index_all_data` via the web interface

3. **Verify indexing**:
   The system will create the following collections:
   - `pharmacy_products` - Pharmaceutical products and health information
   - `metro_rules` - Metro rules, etiquette, and safety information
   - `sports_venues` - Football stadiums and sports venues
   - `sports_content` - Detailed sports team information
   - `ticketing` - Fare and ticketing information
   - `amenities` - Station amenities and nearby services

## Running the Application

### Method 1: Direct Python Execution
```bash
cd repository
python main.py
```

### Method 2: Using uvicorn directly
```bash
cd repository
uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```

The application will start on `http://localhost:3000`

## API Endpoints

### Chat with the Assistant
```bash
POST /chat
Content-Type: application/json

{
  "question": "How do I get to KAFD station?",
  "userid": "user123",
  "lang": "en"
}
```

### Index All Data
```bash
POST /index_all_data
```

### Index Specific Data File
```bash
POST /push_to_vdb?data_path=./data/pharmacy_products.json
```

### Get Chat History
```bash
GET /get_chat_history/{userid}
```

## Usage Examples

### Example 1: Metro Navigation
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I get from KAFD station to King Saud University?",
    "userid": "user123",
    "lang": "en"
  }'
```

### Example 2: Sports Venue Information
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Where is Al Hilal stadium and which metro station is closest?",
    "userid": "user123",
    "lang": "en"
  }'
```

### Example 3: Station Amenities
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What restaurants are near KAFD station?",
    "userid": "user123",
    "lang": "en"
  }'
```

## Project Structure

```
pharmacy-assistant-main/
├── repository/
│   ├── data/                    # Data files
│   │   ├── pharmacy_products.json
│   │   ├── KAFD_rules.json
│   │   ├── football_data.json
│   │   ├── alnasr.txt
│   │   ├── ticketing_data.json
│   │   └── amenities.json
│   ├── modules/
│   │   ├── engine.py           # Main bot engine
│   │   ├── prompts.py          # AI prompts
│   │   ├── tools.py            # Search tools
│   │   └── utils.py            # Indexing utilities
│   ├── main.py                 # FastAPI application
│   └── index.html              # Web interface
├── chromadb/                   # Vector database (created automatically)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure your `.env` file contains the correct API key
   - Verify the API key has sufficient credits

2. **Port Already in Use**:
   - Change the port in `main.py` or kill the process using the port
   - Use: `lsof -ti:3000 | xargs kill -9`

3. **Indexing Errors**:
   - Check that all data files exist in the `repository/data/` directory
   - Ensure you have write permissions for the `chromadb/` directory

4. **Memory Issues**:
   - The indexing process can be memory-intensive
   - Consider indexing files individually if you encounter memory issues

### Logs

The application provides detailed logging. Check the console output for:
- Indexing progress
- Search queries
- Error messages
- API request logs

## Development

### Adding New Data Sources

1. Add your data file to `repository/data/`
2. Update the `MetroDataIndexer` class in `utils.py`
3. Add appropriate search keywords in `tools.py`
4. Re-index the data

### Customizing Prompts

Edit `repository/modules/prompts.py` to modify the AI assistant's behavior and response format.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
