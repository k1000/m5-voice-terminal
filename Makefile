JS_RUNTIME ?= node

.PHONY: test test-pi-startup test-python-pi-startup

test: test-pi-startup test-python-pi-startup

test-pi-startup:
	PI_OFFLINE=1 $(JS_RUNTIME) scripts/test_pi_programmatic_start.mjs

test-python-pi-startup:
	PI_OFFLINE=1 PI_JS_RUNTIME=$(JS_RUNTIME) python3 scripts/test_python_pi_sdk_invocation.py
