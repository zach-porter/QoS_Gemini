# Satellite Return Link QoS Simulation

This project provides a Python-based simulation environment for analyzing Quality of Service (QoS) packet processing on the return link (User Terminal to Gateway via Satellite). It utilizes Object-Oriented Programming (OOP) principles for modularity and includes an interactive web-based Graphical User Interface (GUI) built with Dash for easy parameter adjustment and results visualization.

## Overview

The simulation models a single User Terminal (UT) generating packets belonging to different service classes (Network Control, Expedited Forwarding, Assured Forwarding, Best Effort). These packets are queued at the UT based on their class. A scheduler then selects packets for transmission over a simulated satellite link with configurable bandwidth (static or dynamic). The simulation tracks queue lengths, packet drops, bandwidth utilization, and packet latency to provide insights into the performance of the QoS mechanisms under different conditions.

## Features

*   **Object-Oriented Design:** Core components (Packets, Queues, User Terminal, Scheduler, Simulation) are implemented as distinct classes for better organization and extensibility.
*   **Modular Structure:** Code is organized into separate files within `sim/` (simulation logic) and `gui/` (Dash application) directories.
*   **QoS Service Classes:** Simulates four common QoS classes with strict priority and minimum bandwidth guarantees:
    *   Network Control (NC) - Highest Priority
    *   Expedited Forwarding (EF)
    *   Assured Forwarding (AF)
    *   Best Effort (BE) - Lowest Priority (gets remaining bandwidth)
*   **Configurable Packet Generation:**
    *   Adjustable average packet arrival rate (Poisson process).
    *   Configurable average packet size and standard deviation (Normal distribution).
    *   Defined distribution of packets across service classes (currently set in `sim/config.py`).
*   **Configurable Bandwidth:**
    *   **Static:** Constant bandwidth throughout the simulation.
    *   **Dynamic:** Bandwidth varies over time based on a sine wave (configurable base, amplitude, and period).
*   **Configurable QoS Scheduling:**
    *   Adjustable minimum bandwidth percentage thresholds guaranteed for NC, EF, and AF classes.
*   **Interactive GUI (Dash):**
    *   Web-based interface to easily modify simulation parameters without code changes.
    *   Real-time visualization of results:
        *   Queue lengths (packets) over time for each service class.
        *   Transmitted data (bytes) per time step, stacked by service class.
        *   Available vs. Used bandwidth over time.
        *   Summary statistics (total generated/transmitted/dropped packets, average latency per class, average bandwidth utilization).
        *   Gauge indicators for key performance metrics (e.g., Avg. BW Utilization, Avg. EF Latency, BE Drop Rate).
*   **Command-Line Interface (Optional):** Run simulations directly from the terminal using `main_cli.py` for scripting or non-GUI use cases.

## Project Structure
satellite_qos_sim/
├── sim/ # Core simulation logic
│ ├── init.py
│ ├── packet.py # Packet class definition
│ ├── service_class.py # ServiceClass enum definition
│ ├── queues.py # QoS Queue management class
│ ├── user_terminal.py # User Terminal class (packet generation, queuing)
│ ├── scheduler.py # Satellite Scheduler class (QoS logic)
│ ├── simulation.py # Main simulation orchestration class
│ └── config.py # Default configuration settings
├── gui/ # Dash GUI application
│ ├── init.py
│ ├── app.py # Main Dash application script
│ └── assets/ # Optional: For CSS styling
│ └── styles.css
├── main_cli.py # Optional: Script to run simulation via command line
├── requirements.txt # Python package dependencies
└── README.md # This file

## Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher recommended.
    *   `pip` (Python package installer).

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd satellite_qos_sim
    ```
    Or download and extract the source code ZIP file.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Navigate to the project root directory (`satellite_qos_sim`) in your terminal (where `requirements.txt` is located) and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the GUI Application

1.  Make sure your virtual environment is activated.
2.  Navigate to the **project root directory** (`satellite_qos_sim`) in your terminal.
3.  Run the Dash application:
    ```bash
    python gui/app.py
    ```
4.  Open your web browser and go to: `http://127.0.0.1:8050/` (or the address provided in the terminal output).
5.  Adjust the parameters in the left-hand panel.
6.  Click the "Run Simulation" button.
7.  Observe the results updated in the graphs and summary section on the right.

### Running the Command-Line Simulation (Optional)

1.  Make sure your virtual environment is activated.
2.  Navigate to the **project root directory** (`satellite_qos_sim`) in your terminal.
3.  Run the CLI script:
    ```bash
    python main_cli.py
    ```
4.  The simulation will run using the default (or modified `main_cli.py`) configuration, and the results summary will be printed to the console. You can uncomment the line in `main_cli.py` to save full results to a CSV file.

## GUI Parameters Explained

*   **Simulation Duration (seconds):** Total time the simulation will run.
*   **Time Step (seconds):** Duration of each discrete step in the simulation. Smaller steps increase accuracy but also computation time.
*   **Packet Generation Rate (packets/sec):** Average total number of packets generated by the User Terminal per second across all service classes.
*   **Avg. Packet Size (bytes):** Mean size of generated packets.
*   **Packet Size Std Dev (bytes):** Standard deviation for the packet size distribution.
*   **Max Queue Size (packets per class):** The maximum number of packets allowed in *each individual* service class queue before packets are dropped.
*   **Bandwidth Type:**
    *   `Static`: Uses the "Static Bandwidth" value.
    *   `Dynamic`: Uses the "Base", "Amplitude", and "Period" values to create a varying bandwidth.
*   **Static Bandwidth (Kbps):** The constant bandwidth available if "Static" is selected.
*   **Base Bandwidth (Kbps):** The central value around which dynamic bandwidth oscillates.
*   **Amplitude (Kbps):** The maximum deviation from the base bandwidth for the sine wave.
*   **Period (seconds):** The time it takes for the dynamic bandwidth sine wave to complete one cycle.
*   **QoS Thresholds (% Guaranteed Bandwidth):**
    *   `NC (%)`, `EF (%)`, `AF (%)`: The minimum percentage of the *total available bandwidth* guaranteed to these classes in each time step. They can use more if available and higher-priority classes don't need it. The scheduler processes them in NC -> EF -> AF order.
    *   `BE (%)`: Display-only, shows the percentage remaining after accounting for NC, EF, and AF guarantees (100% - NC% - EF% - AF%). Best Effort traffic uses this remaining bandwidth (and any unused guaranteed bandwidth from other classes).

## Simulation Components Overview

*   **`sim/packet.py`**: Defines the `Packet` class, holding information like ID, service class, size, and arrival/departure times.
*   **`sim/service_class.py`**: Defines the `ServiceClass` enumeration and their inherent priority order.
*   **`sim/queues.py`**: Implements `QoSQueues`, managing separate FIFO queues for each service class within the UT, including drop logic.
*   **`sim/user_terminal.py`**: Models the UT, responsible for generating packets according to configured distributions and placing them into its `QoSQueues`.
*   **`sim/scheduler.py`**: Implements the `SatelliteScheduler`, which contains the core QoS logic for selecting packets from the UT's queues based on priority, available bandwidth, and configured thresholds.
*   **`sim/simulation.py`**: The main `Simulation` class that orchestrates the simulation run, manages time steps, coordinates the UT and Scheduler, calculates bandwidth, and collects results.
*   **`sim/config.py`**: Stores default configuration parameters for the simulation.

## Future Work / Potential Enhancements

*   Simulate multiple User Terminals competing for bandwidth.
*   Implement different scheduling algorithms (e.g., Weighted Fair Queuing - WFQ).
*   Model propagation delay between UT and Gateway.
*   Introduce packet errors or loss based on link conditions.
*   Allow configuration of service class distribution via the GUI.
*   Load/Save simulation configurations from/to files.
*   More sophisticated dynamic bandwidth models (e.g., based on weather, traffic load).
*   More detailed latency analysis (e.g., jitter, histograms).
