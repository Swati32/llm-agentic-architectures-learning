from techniques import corrective, hybrid, hyde, naive, query_decomposition, reranked

ARCHITECTURES = {
    "naive": naive,
    "hybrid": hybrid,
    "reranked": reranked,
    "hyde": hyde,
    "query_decomposition": query_decomposition,
    "corrective": corrective,
}
