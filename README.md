# pytelemsys

`pytelemsys` is a Python package designed to simplify the management, processing, and analysis of telemetry and tracking data. It provides a robust set of tools and utilities to streamline data workflows, enabling efficient handling, detailed analysis, and intuitive visualization of telemetry information.

## Features

The package is organized into the following modules:
- **`pytelem`**: Focused on telemetry data processing and analysis.
- **`pytrack`**: Dedicated to handling and analyzing tracking data.
- **`pyfastf1`** (optional): Loads Formula 1 telemetry through [FastF1](https://github.com/theOehrly/Fast-F1).

## Installation

To install `pytelemsys` directly from GitHub, without cloning the repository:
```bash
pip install "pytelemsys @ git+https://github.com/GiacomoCorradini/pytelemsys.git"
```
To install a specific release, append its tag (e.g. `...pytelemsys.git@v0.1.0`). For the FastF1 extension use `"pytelemsys[fastf1] @ git+https://github.com/GiacomoCorradini/pytelemsys.git"`.

Alternatively, to install it from a local clone, follow these steps:

1. Clone the repository and navigate to the project folder:
    ```bash
    git clone https://github.com/GiacomoCorradini/pytelemsys.git
    cd pytelemsys
    ```
2. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

3. (Optional) To use the FastF1 extension, install with the fastf1 extra:
    ```bash
    pip install -e ".[fastf1]"
    ```

## Example Usage

To get started with `pytelemsys`, you can refer to the example usage provided for each module:
- For telemetry data, check out the [pytelem example](example/test_pytelem.py).
- For tracking data, refer to the [pytrack example](example/test_pytrack.py).
- For Formula 1 data, see the [pyfastf1 example](example/test_pyfastf1.py).
