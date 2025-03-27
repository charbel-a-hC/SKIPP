.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo "  env		prepare environment and install required dependencies"
	@echo "  clean		remove all temp files along with docker images and docker-compose networks"
	@echo "  format	reformat code"
	@echo ""
	@echo "Check the Makefile to know exactly what each target is doing."


.PHONY: env
env:
	which poetry | grep . && echo 'poetry installed' || curl -sSL https://install.python-poetry.org | python3.9 -
	poetry --version
	poetry env use python3.9
	$(eval VIRTUAL_ENVS_PATH=$(shell poetry env info --path))
	@echo $(VIRTUAL_ENVS_PATH)
	poetry install --without dev --no-interaction --no-ansi
	poetry run poe force-cuda11
	poetry show


.PHONY: env-docker
env-docker:
	poetry --version
	poetry run pip install -U pip
	poetry install --without dev --no-interaction --no-ansi
	poetry run poe force-cuda11
	

.PHONY: clean
clean: # Remove Python file artifacts
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -fr {} +


.PHONY: clean-wandb
clean-wandb: # remove all wandb runs
	sudo rm -rf wandb/*
