#!/bin/bash

# assume antrl4 and its Python3 runtime is installed
antlr4='java -jar /usr/local/lib/antlr-4.10.1-complete.jar'
pushd ./ids_embed/parser
	${antlr4} -Dlanguage=Python3 -visitor -o ./antlr ids.g4
popd
