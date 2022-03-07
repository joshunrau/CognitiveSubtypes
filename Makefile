.ONESHELL:
.PHONY: install clean

SHELL:=/bin/bash
PYTHON = venv/bin/python3.9

all: install clean

install: venv
	$(PYTHON) -m pip install .

venv:
	if command -v module &> /dev/null; then 
		module load python/3.9 scipy-stack/2022a
		virtualenv --no-download venv
		$(PYTHON) -m pip install --require-virtualenv --no-index --upgrade pip
		$(PYTHON) -m pip install --require-virtualenv --no-index -r requirements.txt
	else
		virtualenv --no-download venv
		$(PYTHON) -m pip install --require-virtualenv --no-index --upgrade pip
		$(PYTHON) -m pip install notebook
	fi

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	