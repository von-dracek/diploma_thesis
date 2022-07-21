# https://tech.davis-hansson.com/p/make/
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >


# Default - top level rule is what gets ran when you run just `make`
help:
> @echo "make init-setup"
> @echo "	initial setup of the repository for development convenience. should"
> @echo "	be executed just once when repository is clonned"
> @echo "make check"
> @echo "	run checks on the codebase"
> @echo "make install-deps"
> @echo "	install project dependencies"
> @echo "make install-ci-deps"
> @echo "	install CI project dependencies"
> @echo "make update-deps"
> @echo "	update requirements*.txt files after adding dependencies"
.PHONY: help

init-setup:
> ./scripts/init-setup.sh
.PHONY: init-setup

check:
> ./scripts/check.sh
.PHONY: lint

install-deps:
> ./scripts/install-deps.sh
.PHONY: install-deps

install-ci-deps:
> ./scripts/install-ci-deps.sh
.PHONY: install-ci-deps

update-deps:
> ./scripts/update-deps.sh
.PHONY: update-deps
