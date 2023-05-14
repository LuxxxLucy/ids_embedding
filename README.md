# ids_embed

An Embeddings for Ideographic Description Sequence (IDS)

## dependency

1. `antlr4`: use to generate the ANTLR4 parser, based on `ids_embed/parse/ids.g4`
2. `pytorch`:
Others please just refer to `requirements.txt`

## Steps

1. make sure `assets/kanjivg.eids` exists.
2. run `script/prepare.sh` it mainly uses **ANTLR** to generate the parsing code. (mainly calling this command `antlr4 -Dlanguage=Python3 -visitor -o ./antlr ids.g4`)
3.  Train the embedding model to find similar words.
```
./main.py --runner ids_embedding_runner --config ids_embedding.yml
```
Note that training is optional. I am also uploading the model file in the repo because it is small anyway (yeah you can really just use a small model)
