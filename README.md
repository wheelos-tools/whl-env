# whl_env

Autonomous driving environments are inherently complex, involving a wide array of components such as hardware (compute units, vehicles, sensors) and software (operating systems, middleware, the autonomous driving application itself, configuration files).

Ensuring the **consistency** of these environments is paramount during testing. Environment-specific discrepancies are a frequent cause of strange and difficult-to-diagnose issues in system tests, leading to unreliable results.

`whl_env` is designed precisely to tackle this challenge. It acts as a tool to **capture and document the state of your autonomous driving test environment**, allowing you to verify its configuration and guarantee consistency across different test platforms and runs.

## Quick Start

Get `whl_env` installed and start collecting environment data in minutes.

### Installation

Install the package using pip:

```bash
pip install whl_env
```

### Collecting Environment Information

Run the `whl_env` command. Use the `-o` or `--output` option to specify the path where the collected system information should be saved in JSON format.

```bash
whl_env -o system_info.json
```

This command will execute the environment collection process and save the results to `system_info.json` in the current directory.

### Verifying Consistency

Once you have collected `system_info.json` files from different environments or test runs, you can compare them using any standard diff tool (command-line `diff`, graphical diff tools, or dedicated JSON diff comparison tools).

```bash
# Example using a command-line diff tool
diff environment_a_info.json environment_b_info.json

# Any differences highlighted indicate inconsistencies between the two environments.
```

By comparing these snapshots, you can quickly pinpoint configuration drift or variations that could be impacting your test results.
