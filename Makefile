export SHELL:=/bin/bash
export SHELLOPTS:=$(if $(SHELLOPTS),$(SHELLOPTS):)pipefail:errexit

.ONESHELL:

.PHONY: test_dataset_worker
test_dataset_worker:
	function tearDown {
		$(MAKE) stop
	}
	trap tearDown EXIT
	$(MAKE) start_DBs_for_testing
	$(MAKE) start_dataset_worker
	docker compose --env-file .env.test exec dataset_worker python3 -m pytest -s --full-trace

.PHONY: test_training_worker
test_training_worker:
	function tearDown {
		$(MAKE) stop
	}
	trap tearDown EXIT
	$(MAKE) start_DBs_for_testing
	$(MAKE) start_training_worker
	docker compose --env-file .env.test exec training_worker python3 -m pytest -s --full-trace

.PHONY: start_DBs_for_testing
start_DBs_for_testing:
	docker compose --env-file .env.test up --build -d rabbitmq postgres

.PHONY: start_dataset_worker
start_dataset_worker:
	docker compose --env-file .env.test up --build -d dataset_worker

.PHONY: stop
stop:
	docker compose --env-file .env.test down

.PHONY: deploy
deploy:
	$(MAKE) start_dataset_worker
