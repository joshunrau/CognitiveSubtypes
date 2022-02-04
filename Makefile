.DEFAULT_GOAL:= clean-install
.PHONY: clean install clean-install

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

install:
	pip install .

clean-install: install clean