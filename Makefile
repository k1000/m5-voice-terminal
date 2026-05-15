JS_RUNTIME ?= node

.PHONY: test test-pi-startup test-python-pi-startup test-rle test-staleness

test: test-pi-startup test-python-pi-startup test-rle test-staleness

test-pi-startup:
	PI_OFFLINE=1 $(JS_RUNTIME) scripts/test_pi_programmatic_start.mjs

test-python-pi-startup:
	PI_OFFLINE=1 PI_JS_RUNTIME=$(JS_RUNTIME) python3 scripts/test_python_pi_sdk_invocation.py

test-rle:
	python3 scripts/test_rle_roundtrip.py

test-staleness:
	python3 scripts/test_staleness_recovery.py
