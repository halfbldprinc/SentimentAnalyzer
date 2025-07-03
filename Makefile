.PHONY: all train run venv install

MODEL_FILES = sentiment_model.joblib tfidf_vectorizer.joblib
VENV_DIR = venv

all: venv check_model run

venv:
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
		$(VENV_DIR)/bin/pip install --upgrade pip; \
		$(VENV_DIR)/bin/pip install -r requirements.txt; \
	else \
		echo "Virtual environment already exists."; \
		$(VENV_DIR)/bin/pip install -r requirements.txt; \
	fi

check_model:
	@if [ ! -f sentiment_model.joblib ] || [ ! -f tfidf_vectorizer.joblib ]; then \
		echo "Model files not found, training the model..."; \
		$(VENV_DIR)/bin/python load.py; \
	else \
		echo "Model files found, skipping training."; \
	fi

run:
	$(VENV_DIR)/bin/python analyzer.py
