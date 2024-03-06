#!/bin/bash

# Verifica si se proporciona el parámetro "metric"
if [ "$1" == "metrics" ]; then
	# Ejecuta el código de Python para métricas
	python code/metrics
else
	# Ejecuta el código de Python por defecto
	python code
fi
