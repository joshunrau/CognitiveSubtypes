.PHONY: data install clean

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: install clean

model: venv
	./scripts/make_model.sh $(DATA_DIR)

data: venv
	./scripts/make_data.sh $(CURRENT_CSV) $(DATA_DIR)

venv:
	./scripts/make_venv.sh $(ROOT_DIR)

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	