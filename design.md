this is a semantic diff program

input two documents
output a "semantic" diff that's represented in some way that makes it easy to understand. mostly we want to determine if two files are essentially the same in terms of ordering of meaningful sentences. modulo a few bits and pieces added in or removed.

split both documents into sentences

use a sentence embedding model to get embeddings for each sentence

use a sentence similarity model to get similarity between each sentence

run the diff algorithm over the matrix of similarities to get a 'path' through it and produce a comparison

since we're dealing with semantic differences we might need some threshold parameters, but ideally the system would tune them itself
