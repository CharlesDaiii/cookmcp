# CookMCP 🍳

A Model Context Protocol (MCP) server for intelligent recipe search and generation, powered by Qdrant vector database and semantic embeddings.

## Features

- **Semantic Recipe Search**: Find recipes using natural language queries with vector similarity search
- **Smart Recipe Generation**: Create structured recipes based on available ingredients with RAG (Retrieval-Augmented Generation)
- **Recipe Scaling**: Automatically adjust ingredient quantities and nutrition for different serving sizes
- **Ingredient Substitutions**: Get intelligent suggestions for dietary restrictions or pantry limitations
- **WikiBooks Integration**: Crawl and index recipes from WikiBooks cookbook
- **Multi-language Support**: Generate recipes in English or Chinese

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  MCP Client     │────▶│  MCP Server  │────▶│   Qdrant    │
│  (Claude, etc)  │     │  (FastMCP)   │     │  Cloud DB   │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Sentence     │
                        │ Transformer  │
                        │ (BGE-M3)     │
                        └──────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Qdrant Cloud account (or self-hosted Qdrant instance)
- MCP-compatible client (e.g., Claude Desktop)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CharlesDaiii/cookmcp.git
cd cookmcp
```

2. Create a virtual environment:
```bash
python -m venv mcp_env
source mcp_env/bin/activate  # On Windows: mcp_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=recipes
EMBED_MODEL=BAAI/bge-m3
```

## Usage

### 1. Data Collection

Crawl recipes from WikiBooks:
```bash
python wikibooks_crawler.py [max_pages]
# Example: python wikibooks_crawler.py 300
```

This creates `recipes_wikibooks.jsonl` with structured recipe data.

### 2. Upload to Qdrant

Index recipes in your vector database:
```bash
python upload_to_qdrant.py
```

### 3. Test Search

Verify your setup:
```bash
python query_test.py
```

### 4. Run MCP Server

Start the MCP server:
```bash
python mcp_qdrant_recipes.py
```

### 5. Configure MCP Client

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "recipes-qdrant": {
      "command": "python",
      "args": ["/path/to/cookmcp/mcp_qdrant_recipes.py"],
      "env": {
        "QDRANT_URL": "your-qdrant-url",
        "QDRANT_API_KEY": "your-api-key"
      }
    }
  }
}
```

## MCP Tools

### `ping()`
Check Qdrant connectivity and collection status.

### `search_recipes(query, top_k=5, cuisine=None, tags=[])`
Semantic search for recipes.

**Example:**
```python
search_recipes("spicy chicken", cuisine="asian", tags=["quick"])
```

### `generate_recipe(ingredients, servings=2, ...)`
Generate a complete recipe with structured output.

**Parameters:**
- `ingredients`: List of available ingredients
- `servings`: Number of servings (default: 2)
- `cuisine`: Cuisine type (optional)
- `dietary`: Dietary restrictions (vegan, vegetarian, gluten-free, etc.)
- `avoid`: Ingredients to avoid
- `max_time_min`: Maximum cooking time
- `equipment`: Available equipment
- `skill`: Skill level (beginner/intermediate/advanced)
- `use_rag`: Use retrieval-augmented generation
- `language`: Output language (en/zh)

### `scale_recipe(recipe, new_servings)`
Adjust recipe quantities for different serving sizes.

### `suggest_substitutions(recipe, pantry=[], avoid=[])`
Get ingredient substitution suggestions.

## Project Structure

```
cookmcp/
├── mcp_qdrant_recipes.py    # Main MCP server
├── wikibooks_crawler.py      # Recipe data crawler
├── upload_to_qdrant.py       # Vector database indexer
├── query_test.py             # Search testing utility
├── recipes_wikibooks.jsonl   # Crawled recipe data
├── .env                      # Environment configuration
└── README.md                 # This file
```

## Recipe Data Format

```json
{
  "id": "wikibooks:Cookbook:Recipe_Name",
  "title": "Recipe Name",
  "source_url": "https://en.wikibooks.org/wiki/...",
  "cuisine": "Italian",
  "tags": ["quick", "vegetarian"],
  "ingredients": ["ingredient 1", "ingredient 2"],
  "steps": ["Step 1", "Step 2"],
  "servings": 4,
  "total_time_min": 30,
  "license": "CC BY-SA 3.0"
}
```

## Embedding Model

The project uses **BAAI/bge-m3** by default, a multilingual embedding model with:
- 1024-dimensional vectors
- Support for 100+ languages
- Optimized for semantic similarity
- Normalized embeddings for cosine distance

Alternative models can be configured via `EMBED_MODEL` environment variable.

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Debugging
Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Extending the Crawler
To add new recipe sources, implement a crawler following the pattern in `wikibooks_crawler.py`:
1. Fetch recipe data from source
2. Parse into structured format
3. Output as JSONL
4. Index with `upload_to_qdrant.py`

## Requirements

```txt
fastmcp>=0.1.0
qdrant-client>=1.7.0
sentence-transformers>=2.2.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
mwparserfromhell>=0.6.0
requests>=2.31.0
```

## License

This project is licensed under the MIT License. Recipe data from WikiBooks is under CC BY-SA 3.0.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Troubleshooting

### Connection Issues
- Verify Qdrant URL and API key
- Check network connectivity
- Ensure collection exists

### Embedding Errors
- Confirm model is downloaded
- Check CUDA availability for GPU acceleration
- Verify input text encoding

### MCP Integration
- Restart Claude Desktop after config changes
- Check server logs for errors
- Verify Python path in configuration

## Acknowledgments

- WikiBooks Cookbook community for recipe data
- Qdrant team for vector database
- Anthropic for MCP framework
- Hugging Face for embedding models

## Contact

For questions or support, please open an issue on GitHub.
