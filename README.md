# CLI Search Engine 🔍

A tiny search engine core with inverted index and TF-IDF ranking — just like what powers Google, but local!

## Features

- **Inverted Index**: Maps terms to documents for O(1) term lookups
- **TF-IDF Scoring**: Ranks results by relevance using Term Frequency-Inverse Document Frequency
- **BM25 Support**: Modern ranking algorithm (optional)
- **Smart Tokenization**: Handles stop words, normalization, and case folding
- **Snippet Generation**: Shows relevant document excerpts
- **Incremental Indexing**: Add documents to existing index
- **Persistent Storage**: Index saved to disk for fast reloads

## Quick Start

```bash
# Index a directory
python search.py index ./my-documents

# Search for documents
python search.py query "cloudflare worker"

# View index statistics
python search.py stats

# Clear the index
python search.py clear
```

## Installation

```bash
# Make it globally accessible
ln -s $(pwd)/search.py ~/.local/bin/search

# Or use directly
python search.py --help
```

## Commands

### `index <directory>`

Index all documents in a directory.

```bash
# Index all text files
python search.py index ./docs

# Index only Python files
python search.py index . --pattern "*.py"

# Index multiple patterns
python search.py index . --pattern "*.md" --pattern "*.txt"

# Exclude certain patterns
python search.py index . --exclude "node_modules/*" --exclude "*.min.js"

# Verbose output
python search.py index ./docs --verbose
```

### `query "search terms"`

Search the indexed documents.

```bash
# Basic search
python search.py query "cloudflare workers"

# Limit results
python search.py query "python" --limit 5

# Use BM25 ranking (often better for short queries)
python search.py query "api" --algorithm bm25
```

### `stats`

Show index statistics.

```bash
python search.py stats
```

Output:
```
========================================
Index Statistics
========================================
  Total Documents: 1,245
  Unique Terms: 45,230
  Average Doc Length: 125.5 words
  Index Size: 2.35 MB
```

### `clear`

Clear the entire index (requires confirmation).

```bash
python search.py clear
```

## How It Works

### 1. Tokenization

Documents are broken into tokens:
- Lowercased
- Punctuation removed
- Stop words filtered (the, and, is, etc.)
- Minimum 2 characters

### 2. Inverted Index

```
Term          Documents
----          ---------
cloudflare    {doc1: 5, doc3: 2, doc7: 1}
worker        {doc1: 3, doc5: 4}
api           {doc2: 8, doc3: 3, doc5: 2}
```

### 3. TF-IDF Scoring

For each term in a document:
```
TF  = term_frequency / document_length
IDF = log(total_docs / docs_containing_term)
Score = TF × IDF
```

Higher scores for:
- Terms that appear frequently in a document
- Terms that appear in few documents overall (more distinctive)

### 4. BM25 (Optional)

Modern variant that handles:
- Document length normalization (better for varying doc sizes)
- Term saturation (prevents over-weighting very frequent terms)

## Architecture

```
┌─────────────────┐
│  CLI Interface  │
└────────┬────────┘
         │
┌────────▼────────┐
│  SearchEngine   │
│  - search()     │
│  - index_dir()  │
└────────┬────────┘
         │
┌────────▼────────┐     ┌─────────────┐
│ InvertedIndex   │────▶│  index.pkl  │
│  - index        │     │  (binary)   │
│  - doc_freq     │     └─────────────┘
│  - term_freq    │
│  - documents    │────▶┌─────────────┐
└─────────────────┘     │metadata.json│
                        │  (human)    │
                        └─────────────┘
```

## Supported File Types

By default, these file types are indexed:
- `.txt`, `.md`, `.rst` - Text documents
- `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs` - Code
- `.html`, `.css` - Web
- `.json`, `.yaml`, `.yml` - Config
- And more...

## Performance

| Metric | Typical |
|--------|---------|
| Indexing Speed | ~1000 docs/sec |
| Query Time | <10ms for 10k docs |
| Memory Usage | ~2x raw text size |

## Example Session

```bash
# Create some test documents
mkdir test_docs
echo "Cloudflare Workers are serverless functions that run on Cloudflare's edge network." > test_docs/cf-workers.txt
echo "AWS Lambda is a serverless compute service." > test_docs/aws-lambda.txt
echo "Python is a programming language." > test_docs/python.txt

# Index them
$ python search.py index test_docs
Found 3 files to index
Indexing complete!
  Successfully indexed: 3 files
  Total terms in index: 15

# Search
$ python search.py query "serverless"
============================================================
Search Results for: 'serverless'
============================================================
Found 2 result(s)

  1. Cloudflare Workers are serverless functions that run...
     📄 cf-workers.txt
     ⭐ Score: 0.2310
     📝 Cloudflare Workers are serverless functions that run on...
     🔍 'serverless': 1

  2. AWS Lambda is a serverless compute service.
     📄 aws-lambda.txt
     ⭐ Score: 0.2310
     📝 AWS Lambda is a serverless compute service.
     🔍 'serverless': 1

# Search for multiple terms
$ python search.py query "cloudflare edge"
============================================================
Search Results for: 'cloudflare edge'
============================================================
Found 1 result(s)

  1. Cloudflare Workers are serverless functions...
     📄 cf-workers.txt
     ⭐ Score: 0.4621
     📝 Cloudflare Workers are serverless functions that run on Cloudflare's edge...
     🔍 'cloudflare': 1, 'edge': 1

# Check stats
$ python search.py stats
========================================
Index Statistics
========================================
  Total Documents: 3
  Unique Terms: 15
  Average Doc Length: 10.33 words
  Index Size: 0.0 MB
```

## Extending

### Custom Tokenizer

```python
from search import Tokenizer

tokenizer = Tokenizer(
    lowercase=True,
    remove_stopwords=True
)
tokens = tokenizer.tokenize("Your text here")
```

### Programmatic API

```python
from search import SearchEngine

engine = SearchEngine()

# Index
engine.index_directory("./docs")

# Search
results = engine.search("cloudflare workers", top_k=5)
for r in results:
    print(f"{r.doc_id}: {r.score}")
```

## Storage Format

- **index.pkl**: Binary pickle of the inverted index (fast loading)
- **metadata.json**: Human-readable document metadata

## License

MIT
