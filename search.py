#!/usr/bin/env python3
"""
CLI Search Engine - Local Inverted Index
A tiny search engine core with TF-IDF ranking.

Usage:
    search index ./documents       # Index a directory
    search query "cloudflare worker"      # Search the index
    search stats                   # Show index statistics
    search clear                   # Clear the index
"""

import os
import sys
import json
import re
import math
import pickle
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import fnmatch


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""
    term_freq: Dict[str, int] = None
    
    def __post_init__(self):
        if self.term_freq is None:
            self.term_freq = {}


class Tokenizer:
    """Handles text tokenization and normalization."""
    
    # Common English stop words to filter out
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
        'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
        'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what',
        'when', 'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'just', 'should', 'now', 'id', 'co', 'etc', 'eg', 'ie'
    }
    
    def __init__(self, lowercase: bool = True, remove_stopwords: bool = True):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual terms.
        
        - Converts to lowercase
        - Removes punctuation
        - Filters stop words
        - Minimum length of 2 characters
        """
        if self.lowercase:
            text = text.lower()
        
        # Extract words (alphanumeric sequences)
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        
        # Filter stop words and short tokens
        if self.remove_stopwords:
            tokens = [
                token for token in tokens 
                if token not in self.STOP_WORDS and len(token) >= 2
            ]
        else:
            tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def extract_title(self, text: str, max_lines: int = 5) -> str:
        """Extract a title from the beginning of the document."""
        lines = text.split('\n')[:max_lines]
        for line in lines:
            line = line.strip()
            if line and len(line) < 200:
                return line[:100]
        return "Untitled"
    
    def extract_snippet(self, text: str, query_terms: List[str], 
                        max_length: int = 150) -> str:
        """Extract a relevant snippet containing query terms."""
        text_lower = text.lower()
        best_pos = 0
        best_score = 0
        
        # Find best position based on query term density
        for i in range(0, len(text) - max_length, 50):
            window = text_lower[i:i + max_length]
            score = sum(1 for term in query_terms if term.lower() in window)
            if score > best_score:
                best_score = score
                best_pos = i
        
        snippet = text[best_pos:best_pos + max_length].strip()
        # Clean up snippet
        snippet = re.sub(r'\s+', ' ', snippet)
        
        if len(snippet) == max_length:
            snippet = snippet.rsplit(' ', 1)[0] + '...'
        
        return snippet


class InvertedIndex:
    """
    Inverted index data structure for efficient text search.
    
    Maps terms to documents they appear in, with frequency information.
    """
    
    def __init__(self, index_dir: str = ".search_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Core inverted index: term -> {doc_id: frequency}
        self.index: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Document metadata: doc_id -> {path, length, title}
        self.documents: Dict[str, Dict] = {}
        
        # Document frequencies: term -> number of docs containing term
        self.doc_freq: Dict[str, int] = defaultdict(int)
        
        # Term frequencies per document: doc_id -> {term: frequency}
        self.term_freq: Dict[str, Dict[str, int]] = defaultdict(Counter)
        
        # Total documents in index
        self.total_docs = 0
        
        # Index version for compatibility
        self.version = "1.0"
        
        self._load_index()
    
    def _index_path(self) -> Path:
        return self.index_dir / "index.pkl"
    
    def _metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"
    
    def _load_index(self):
        """Load index from disk if it exists."""
        index_file = self._index_path()
        meta_file = self._metadata_path()
        
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.index = defaultdict(dict, data.get('index', {}))
                    self.doc_freq = defaultdict(int, data.get('doc_freq', {}))
                    self.term_freq = defaultdict(Counter, 
                        {k: Counter(v) for k, v in data.get('term_freq', {}).items()})
            except Exception as e:
                print(f"Warning: Could not load index: {e}")
        
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    self.documents = meta.get('documents', {})
                    self.total_docs = meta.get('total_docs', 0)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
    
    def save(self):
        """Save index to disk."""
        # Save main index as pickle (faster for large data)
        index_data = {
            'index': dict(self.index),
            'doc_freq': dict(self.doc_freq),
            'term_freq': {k: dict(v) for k, v in self.term_freq.items()},
            'version': self.version
        }
        
        with open(self._index_path(), 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata as JSON (human readable)
        meta_data = {
            'documents': self.documents,
            'total_docs': self.total_docs,
            'version': self.version,
            'num_terms': len(self.index)
        }
        
        with open(self._metadata_path(), 'w') as f:
            json.dump(meta_data, f, indent=2)
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            text: Document content
            metadata: Additional document metadata
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text)
        
        if not tokens:
            return
        
        # Count term frequencies
        token_counts = Counter(tokens)
        
        # Update inverted index
        for term, freq in token_counts.items():
            self.index[term][doc_id] = freq
            self.doc_freq[term] = len(self.index[term])
        
        # Store term frequencies for this document
        self.term_freq[doc_id] = token_counts
        
        # Store document metadata
        self.documents[doc_id] = {
            'id': doc_id,
            'length': len(tokens),
            'unique_terms': len(token_counts),
            'title': tokenizer.extract_title(text),
            **(metadata or {})
        }
        
        self.total_docs = len(self.documents)
    
    def remove_document(self, doc_id: str):
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return
        
        # Remove from inverted index
        for term in list(self.index.keys()):
            if doc_id in self.index[term]:
                del self.index[term][doc_id]
                self.doc_freq[term] = len(self.index[term])
                if not self.index[term]:
                    del self.index[term]
                    del self.doc_freq[term]
        
        # Remove from term frequencies
        if doc_id in self.term_freq:
            del self.term_freq[doc_id]
        
        # Remove from documents
        del self.documents[doc_id]
        self.total_docs = len(self.documents)
    
    def clear(self):
        """Clear the entire index."""
        self.index.clear()
        self.documents.clear()
        self.doc_freq.clear()
        self.term_freq.clear()
        self.total_docs = 0
        
        # Delete files
        if self._index_path().exists():
            self._index_path().unlink()
        if self._metadata_path().exists():
            self._metadata_path().unlink()
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'total_documents': self.total_docs,
            'total_terms': len(self.index),
            'index_size_mb': round(
                (self._index_path().stat().st_size if self._index_path().exists() else 0) / 1024 / 1024, 2
            ),
            'avg_doc_length': round(
                sum(d.get('length', 0) for d in self.documents.values()) / max(self.total_docs, 1), 2
            )
        }


class SearchEngine:
    """
    Full-text search engine with TF-IDF ranking.
    """
    
    def __init__(self, index_dir: str = ".search_index"):
        self.index = InvertedIndex(index_dir)
        self.tokenizer = Tokenizer()
    
    def _compute_tf_idf(self, term: str, doc_id: str) -> float:
        """
        Compute TF-IDF score for a term in a document.
        
        TF = term frequency / document length
        IDF = log(total_docs / document_frequency)
        """
        if doc_id not in self.index.term_freq or term not in self.index.term_freq[doc_id]:
            return 0.0
        
        # Term Frequency (normalized by document length)
        tf = self.index.term_freq[doc_id][term]
        doc_length = self.index.documents[doc_id].get('length', 1)
        normalized_tf = tf / doc_length if doc_length > 0 else 0
        
        # Inverse Document Frequency
        doc_freq = self.index.doc_freq.get(term, 1)
        idf = math.log(self.index.total_docs / doc_freq) if doc_freq > 0 else 0
        
        return normalized_tf * idf
    
    def _compute_bm25(self, term: str, doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Compute BM25 score (improved TF-IDF variant used by modern search engines).
        
        BM25 often performs better than raw TF-IDF for short queries.
        """
        if doc_id not in self.index.term_freq or term not in self.index.term_freq[doc_id]:
            return 0.0
        
        tf = self.index.term_freq[doc_id][term]
        doc_length = self.index.documents[doc_id].get('length', 1)
        avg_doc_length = sum(
            d.get('length', 1) for d in self.index.documents.values()
        ) / max(self.index.total_docs, 1)
        
        doc_freq = self.index.doc_freq.get(term, 1)
        idf = math.log(
            (self.index.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1
        )
        
        # BM25 formula
        score = idf * (
            (tf * (k1 + 1)) / 
            (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
        )
        
        return score
    
    def search(self, query: str, top_k: int = 10, 
               algorithm: str = "tfidf") -> List[SearchResult]:
        """
        Search the index for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            algorithm: Scoring algorithm ("tfidf" or "bm25")
        
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if self.index.total_docs == 0:
            return []
        
        # Tokenize query
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return []
        
        # Find candidate documents (union of docs containing any query term)
        candidate_docs: Set[str] = set()
        for term in query_terms:
            candidate_docs.update(self.index.index.get(term, {}).keys())
        
        # Score each candidate document
        scores: Dict[str, float] = {}
        term_matches: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        score_func = self._compute_bm25 if algorithm == "bm25" else self._compute_tf_idf
        
        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                term_score = score_func(term, doc_id)
                score += term_score
                if term_score > 0:
                    term_matches[doc_id][term] = self.index.term_freq[doc_id].get(term, 0)
            
            # Boost exact phrase matches
            if len(query_terms) > 1:
                doc_text = self._get_doc_text(doc_id)
                query_lower = query.lower()
                if query_lower in doc_text.lower():
                    score *= 2.0  # Significant boost for exact matches
            
            scores[doc_id] = score
        
        # Sort by score (descending)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in ranked_docs[:top_k]:
            if score > 0:
                doc_meta = self.index.documents.get(doc_id, {})
                
                # Get document text for snippet
                doc_text = self._get_doc_text(doc_id)
                snippet = self.tokenizer.extract_snippet(doc_text, query_terms)
                
                result = SearchResult(
                    doc_id=doc_id,
                    score=round(score, 4),
                    title=doc_meta.get('title', 'Untitled'),
                    snippet=snippet,
                    term_freq=dict(term_matches[doc_id])
                )
                results.append(result)
        
        return results
    
    def _get_doc_text(self, doc_id: str) -> str:
        """Retrieve original document text if available."""
        # Try to read from stored path
        doc_meta = self.index.documents.get(doc_id, {})
        if 'path' in doc_meta:
            path = Path(doc_meta['path'])
            if path.exists():
                try:
                    return path.read_text(encoding='utf-8', errors='ignore')
                except:
                    pass
        return ""
    
    def index_directory(self, directory: str, 
                        patterns: List[str] = None,
                        exclude_patterns: List[str] = None,
                        verbose: bool = True):
        """
        Index all files in a directory.
        
        Args:
            directory: Path to directory to index
            patterns: File patterns to include (e.g., ['*.txt', '*.md'])
            exclude_patterns: Patterns to exclude
            verbose: Print progress
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Default patterns: common text files
        if patterns is None:
            patterns = ['*.txt', '*.md', '*.rst', '*.py', '*.js', '*.ts', 
                       '*.html', '*.css', '*.json', '*.yaml', '*.yml',
                       '*.java', '*.go', '*.rs', '*.c', '*.cpp', '*.h',
                       '*.rb', '*.php', '*.swift', '*.kt', '*.scala']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '*.min.js', '*.min.css', 'node_modules/*', '.git/*',
                '__pycache__/*', '*.pyc', '.venv/*', 'venv/*',
                '.search_index/*', 'dist/*', 'build/*'
            ]
        
        # Collect files to index
        files_to_index = []
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                # Check exclude patterns
                relative_path = file_path.relative_to(directory)
                should_exclude = any(
                    fnmatch.fnmatch(str(relative_path), exclude) or
                    any(fnmatch.fnmatch(str(p), exclude.rstrip('/*')) 
                        for p in relative_path.parents)
                    for exclude in exclude_patterns
                )
                
                if not should_exclude and file_path.is_file():
                    files_to_index.append(file_path)
        
        if verbose:
            print(f"Found {len(files_to_index)} files to index")
        
        # Index files
        indexed_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(files_to_index, 1):
            if verbose and i % 100 == 0:
                print(f"  Indexed {i}/{len(files_to_index)} files...")
            
            try:
                # Create document ID from relative path
                doc_id = str(file_path.relative_to(directory))
                
                # Remove old version if exists
                if doc_id in self.index.documents:
                    self.index.remove_document(doc_id)
                
                # Read and index file
                text = file_path.read_text(encoding='utf-8', errors='ignore')
                
                metadata = {
                    'path': str(file_path),
                    'filename': file_path.name,
                    'extension': file_path.suffix,
                    'size_bytes': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime
                }
                
                self.index.add_document(doc_id, text, metadata)
                indexed_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"  Failed to index {file_path}: {e}")
                failed_count += 1
        
        # Save index
        self.index.save()
        
        if verbose:
            print(f"\nIndexing complete!")
            print(f"  Successfully indexed: {indexed_count} files")
            if failed_count > 0:
                print(f"  Failed: {failed_count} files")
            print(f"  Total terms in index: {len(self.index.index)}")
    
    def get_stats(self) -> Dict:
        """Get search engine statistics."""
        return self.index.get_stats()


def format_results(results: List[SearchResult], query: str) -> str:
    """Format search results for display."""
    if not results:
        return f"\nNo results found for: '{query}'\n"
    
    lines = [
        f"\n{'=' * 60}",
        f"Search Results for: '{query}'",
        f"{'=' * 60}",
        f"Found {len(results)} result(s)\n"
    ]
    
    for i, result in enumerate(results, 1):
        lines.append(f"  {i}. {result.title}")
        lines.append(f"     📄 {result.doc_id}")
        lines.append(f"     ⭐ Score: {result.score}")
        
        if result.snippet:
            # Highlight query terms in snippet
            snippet = result.snippet
            lines.append(f"     📝 {snippet}")
        
        if result.term_freq:
            term_info = ", ".join(
                f"'{term}': {freq}" for term, freq in result.term_freq.items()
            )
            lines.append(f"     🔍 {term_info}")
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="CLI Search Engine - Local Inverted Index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  search index ./documents          Index a directory
  search index . --pattern "*.py"   Index only Python files
  search query "cloudflare worker"  Search for documents
  search stats                      Show index statistics
  search clear                      Clear the index
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('directory', help='Directory to index')
    index_parser.add_argument('--pattern', '-p', action='append',
                             help='File pattern to include (can use multiple)')
    index_parser.add_argument('--exclude', '-e', action='append',
                             help='Pattern to exclude (can use multiple)')
    index_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search the index')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('-n', '--limit', type=int, default=10,
                             help='Maximum results (default: 10)')
    query_parser.add_argument('--algorithm', choices=['tfidf', 'bm25'],
                             default='tfidf',
                             help='Scoring algorithm (default: tfidf)')
    
    # Stats command
    subparsers.add_parser('stats', help='Show index statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear the index')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    engine = SearchEngine()
    
    if args.command == 'index':
        try:
            engine.index_directory(
                args.directory,
                patterns=args.pattern,
                exclude_patterns=args.exclude,
                verbose=args.verbose
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'query':
        results = engine.search(args.query, top_k=args.limit, 
                               algorithm=args.algorithm)
        print(format_results(results, args.query))
    
    elif args.command == 'stats':
        stats = engine.get_stats()
        print("\n" + "=" * 40)
        print("Index Statistics")
        print("=" * 40)
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Unique Terms: {stats['total_terms']:,}")
        print(f"  Average Doc Length: {stats['avg_doc_length']} words")
        print(f"  Index Size: {stats['index_size_mb']} MB")
        print()
    
    elif args.command == 'clear':
        confirm = input("Are you sure you want to clear the index? (yes/no): ")
        if confirm.lower() == 'yes':
            engine.index.clear()
            print("Index cleared successfully.")
        else:
            print("Cancelled.")


if __name__ == '__main__':
    main()
