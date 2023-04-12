.PHONY: docs clean

all: init docs test ci clean test-readme

init:
    # pip install -e .[socks]
	pip install -r requirements-dev.txt

docs:
	cd docs && make html
	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/_build/html/index.html.\n\033[0m"

test:
	# -find coverage/ -mindepth 1 -delete
	pytest $${TESTS}

ci:
	pytest tests --junitxml=report.xml

clean:
	find . -name '*.py[co]' -delete

dist: test
	python setup.py sdist

test-readme:
	python setup.py check --restructuredtext --strict && ([ $$? -eq 0 ] && echo "README.rst ok") || echo "Invalid markup in README.rst!"