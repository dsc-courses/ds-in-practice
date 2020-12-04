
build:
	jupyter-book build ./book

ghpages:
	ghp-import -n -p -f book/_build/html
