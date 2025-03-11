all: consistency derivatives

consistency:
	python consistency.py
	latexmk -pdf table-consistency.tex
	latexmk -c

derivatives:
	python derivatives.py
	latexmk -pdf derivatives-tables.tex
	latexmk -c