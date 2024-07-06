# Database
## Ann vector storage
```json
{
    "name": "ann",
    "args": {
        "router_path": "your service url" ,
        "port": "your service port"
    }
}
```

# Loader
## Text Loader
```json
"loader": {
    "name": "text",
    "args": {}
},
```

# Embedding
## Bge-m3 embedding
```json
"embedding": {
    "name": "bge",
    "args": {
        "router_path": "your service url",
        "port": "your service port"
    }
},
```

# Splitter
## Recursive splitter
```json
"splitter": {
    "name": "recursive",
    "args": {
        "chunk_size": 400
    }
},
```

# Web Search
## Arxiv
```json
{
    "name": "arxiv",
    "args": {}
},
```

## Baidu
```json
{
    "name": "baidu",
    "args": {}
}
```

# Rerank
## Bge normal reranker
```json
"rerank": {
    "name": "bge_normal",
    "args": {
        "router_path": "your service url",
        "port": "your service port",
        "path": "bge_normal_rerank"
    }
}
```

# Prompt generation
## Default generation
```json
"prompt_gen": {
    "name": "base",
    "args": {}
},
```

# Prompt zip
## Llmlingua2
```json
"prompt_zip": {
    "name": "llmlingua2",
    "args": {
        "router_path": "your service url",
        "port": "your service port",
        "path": "llmlingua2"
    }
},
```

# Chat model
## Llama3-Chinese-8B
```json
"response_gen": {
    "name": "llama3",
    "args": {
        "router_path": "your service url",
        "port": "your service port"
    }
}
```