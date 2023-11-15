export SHELL:=/bin/bash
export SHELLOPTS:=$(if $(SHELLOPTS),$(SHELLOPTS):)pipefail:errexit

.ONESHELL:

.PHONY: test_dataset_worker
test_dataset_worker:
	function tearDown {
		docker-compose down
	}
	trap tearDown EXIT
	docker-compose --env-file .env.test up --build -d dataset_worker rabbitmq
	docker-compose exec dataset_worker python3 -m pytest -s
