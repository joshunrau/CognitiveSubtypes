.DEFAULT_GOAL:= clean-install
.PHONY: clean install clean-install

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

install:
	@if [ $${CONDA_PREFIX:(-17):17} != CognitiveSubtypes ]; then echo ERROR: must install in CognitiveSubtypes environment && exit 1; fi
	pip install -r requirements.txt
	pip install .

clean-install: install clean