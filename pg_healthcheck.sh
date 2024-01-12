#!/bin/bash
# pg_healthcheck.sh

pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}

