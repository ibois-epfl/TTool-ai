export SHELL:=/bin/bash
export SHELLOPTS:=$(if $(SHELLOPTS),$(SHELLOPTS):)pipefail:errexit

.ONESHELL:

.PHONY: test_dataset_worker
test_dataset_worker:
	function tearDown {
		docker compose --env-file .env.test down
	}
	trap tearDown EXIT
	docker compose --env-file .env.test up --build -d
	docker compose --env-file .env.test exec dataset_worker python3 -m pytest -s
