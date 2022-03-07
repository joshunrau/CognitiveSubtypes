.ONESHELL:
.PHONY: install clean

PYTHON = venv/bin/python3.9

all: install clean

install: venv
	$(PYTHON) -m pip install .

venv:
	if command -v module &> /dev/null; then module load python/3.9 scipy-stack/2021a; fi
	virtualenv --no-download venv
	$(PYTHON) -m pip install --no-index --upgrade pip

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	