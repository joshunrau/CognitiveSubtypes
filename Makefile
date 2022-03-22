.PHONY: data install clean

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: install clean

train: venv
	./scripts/make_train.sh

data:
	./scripts/make_data.sh $(CURRENT_CSV) $(DATA_DIR)

venv:
	./scripts/make_venv.sh ${ROOT_DIR}

clean:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	