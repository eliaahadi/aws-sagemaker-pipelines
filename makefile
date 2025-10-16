.PHONY: train register approve deploy invoke reset

PY := python

train:
	$(PY) src/train.py

register:
	$(PY) src/register.py

approve:
	# Approve latest version by default; override with VERSION=v0003
	$(PY) src/approve.py --version $${VERSION:-latest}

deploy:
	uvicorn serve.app:app --host 127.0.0.1 --port 8000 --reload

invoke:
	curl -s -X POST http://127.0.0.1:8000/invocations \
		-H "Content-Type: application/json" \
		-d '{"instances": [[5.1,3.5,1.4,0.2],[6.2,3.1,5.1,2.3]]}' | jq .

reset:
	rm -rf artifacts model_registry
	mkdir -p artifacts model_registry