"""Movie search engine implementation."""

from cli.text_processing import normalize_text, tokenize, remove_stopwords, stem_tokens
from cli.data_loader import load_movies, load_stopwords


class MovieSearchEngine:
    """Search engine for finding movies by keyword matching."""

    def __init__(self, movies_filepath: str = "data/movies.json",
                 stopwords_filepath: str = "data/stopwords.txt"):
        """Initialize search engine with data."""
        self.movies = load_movies(movies_filepath)
        self.stopwords = load_stopwords(stopwords_filepath)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search for movies matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of matching movies sorted by ID
        """
        processed_query = self._process_query(query)
        matching_movies = self._find_matches(processed_query)
        return self._sort_and_limit(matching_movies, max_results)

    def _process_query(self, query: str) -> list[str]:
        """Process query text into stemmed tokens."""
        normalized = normalize_text(query)
        tokens = tokenize(normalized)
        filtered_tokens = remove_stopwords(tokens, self.stopwords)
        return stem_tokens(filtered_tokens)

    def _process_title(self, title: str) -> list[str]:
        """Process movie title into tokens."""
        normalized = normalize_text(title)
        return tokenize(normalized)

    def _find_matches(self, query_tokens: list[str]) -> list[dict]:
        """Find movies where any query token appears in the title."""
        results = []

        for movie in self.movies:
            title_tokens = self._process_title(movie["title"])

            if self._has_match(query_tokens, title_tokens):
                results.append(movie)

        return results

    def _has_match(self, query_tokens: list[str], title_tokens: list[str]) -> bool:
        """Check if any query token is found in any title token."""
        return any(
            query_token in title_token
            for query_token in query_tokens
            for title_token in title_tokens
        )

    def _sort_and_limit(self, movies: list[dict], limit: int) -> list[dict]:
        """Sort movies by ID and limit results."""
        sorted_movies = sorted(movies, key=lambda m: m["id"])
        return sorted_movies[:limit]
