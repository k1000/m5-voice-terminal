.PHONY: test test-pi-startup test-python-pi-startup

test: test-pi-startup test-python-pi-startup

test-pi-startup:
	PI_OFFLINE=1 node scripts/test_pi_programmatic_start.mjs

test-python-pi-startup:
	PI_OFFLINE=1 python3 scripts/test_python_pi_sdk_invocation.py
