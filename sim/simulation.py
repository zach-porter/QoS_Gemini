# sim/simulation.py
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from .user_terminal import UserTerminal
from .scheduler import SatelliteScheduler
from .service_class import ServiceClass, PRIORITY_ORDER
from .config import bps_to_bytes_per_step

class Simulation:
    """Orchestrates the satellite QoS simulation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the simulation environment.

        Args:
            config: A dictionary containing all simulation parameters.
        """
        self.config = config
        self.duration = config["simulation_duration_sec"]
        self.time_step = config["time_step_sec"]
        self.current_time = 0.0

        # Initialize components
        # For simplicity, we simulate only one User Terminal
        self.user_terminal = UserTerminal(terminal_id=1, config=config["ut_config"])
        self.scheduler = SatelliteScheduler(config["qos_thresholds_percent"])

        # Bandwidth calculation function
        self.bandwidth_type = config["bandwidth_type"]
        if self.bandwidth_type == "static":
            static_bps = config["static_bandwidth_kbps"] * 1000
            self._static_bw_bytes_per_step = bps_to_bytes_per_step(static_bps, self.time_step)
            self.get_available_bandwidth = self._get_static_bandwidth
        elif self.bandwidth_type == "dynamic":
            self.dyn_bw_config = config["dynamic_bw_config"]
            self.get_available_bandwidth = self._get_dynamic_bandwidth
        else:
            raise ValueError(f"Unknown bandwidth type: {self.bandwidth_type}")

        # Data collection
        self.history = [] # Stores state at each time step
        self.all_transmitted_packets = [] # Stores all packets that were successfully transmitted

    def _get_static_bandwidth(self, current_time: float) -> int:
        """Returns the static bandwidth in bytes for the current time step."""
        return self._static_bw_bytes_per_step

    def _get_dynamic_bandwidth(self, current_time: float) -> int:
        """Calculates dynamic bandwidth based on a sine wave variation."""
        base_bps = self.dyn_bw_config["base_kbps"] * 1000
        amplitude_bps = self.dyn_bw_config["amplitude_kbps"] * 1000
        period = self.dyn_bw_config["period_sec"]

        # Calculate current BW using a sine wave centered around the base
        # Ensure bandwidth doesn't go below a minimum (e.g., 10% of base)
        min_bw_bps = base_bps * 0.1
        current_bw_bps = base_bps + amplitude_bps * math.sin(2 * math.pi * current_time / period)
        current_bw_bps = max(min_bw_bps, current_bw_bps) # Ensure minimum bandwidth

        return bps_to_bytes_per_step(current_bw_bps, self.time_step)

    def run_step(self):
        """Runs a single step of the simulation."""
        # 1. Generate packets at the User Terminal
        self.user_terminal.generate_packets(self.current_time, self.time_step)

        # 2. Determine available bandwidth for this step
        available_bw_bytes = self.get_available_bandwidth(self.current_time)

        # 3. Schedule packets for transmission
        queues = self.user_terminal.get_queues()
        transmitted_packets, transmitted_bytes_per_class = self.scheduler.schedule_packets(
            queues, available_bw_bytes, self.current_time + self.time_step # Assume transmission completes by end of step
        )

        # 4. Collect data for this step
        queue_stats = queues.get_all_queue_stats()
        step_data = {
            "time": self.current_time + self.time_step,
            "available_bw_bytes": available_bw_bytes,
            "total_transmitted_bytes": sum(transmitted_bytes_per_class.values()),
            "bw_utilization_percent": (sum(transmitted_bytes_per_class.values()) * 100.0 / available_bw_bytes) if available_bw_bytes > 0 else 0,
        }
        for sc in PRIORITY_ORDER:
            label = sc.label
            step_data[f"queue_pkts_{label}"] = queue_stats[label]["packets"]
            step_data[f"queue_bytes_{label}"] = queue_stats[label]["bytes"]
            step_data[f"dropped_pkts_{label}"] = queue_stats[label]["dropped"] # Cumulative drops reported by queue
            step_data[f"tx_bytes_{label}"] = transmitted_bytes_per_class.get(sc, 0)
            step_data[f"tx_pkts_{label}"] = sum(1 for p in transmitted_packets if p.service_class == sc)

        self.history.append(step_data)
        self.all_transmitted_packets.extend(transmitted_packets)

        # 5. Advance simulation time
        self.current_time += self.time_step

    def run_simulation(self):
        """Runs the full simulation duration."""
        print(f"Starting simulation: Duration={self.duration}s, Time Step={self.time_step}s")
        start_sim_wall_time = time.time()

        num_steps = int(self.duration / self.time_step)
        for i in range(num_steps):
            self.run_step()
            if (i + 1) % (num_steps // 10) == 0: # Print progress update
                 print(f"  Progress: {int((i + 1) * 100 / num_steps)}% completed...")


        end_sim_wall_time = time.time()
        print(f"Simulation finished in {end_sim_wall_time - start_sim_wall_time:.2f} seconds (wall clock).")

    def get_results_dataframe(self) -> pd.DataFrame:
        """Returns the simulation history as a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame() # Return empty DataFrame if simulation hasn't run
        return pd.DataFrame(self.history)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates summary statistics after the simulation."""
        if not self.history:
            return {"error": "Simulation not run yet."}

        df = self.get_results_dataframe()
        summary = {"total_time": self.current_time}

        total_generated = {}
        total_transmitted = {}
        total_dropped = {}
        total_latency = {}
        avg_latency = {}
        packet_counts = {}

        # Calculate generated packets (approximation based on rate * time)
        # A more accurate way would be to count in UserTerminal, but this is simpler for now
        gen_rate = self.config["ut_config"]["packet_rate_pps"]
        sc_dist = self.config["ut_config"]["service_class_distribution"]
        approx_total_gen = gen_rate * self.duration

        for sc in PRIORITY_ORDER:
            label = sc.label
            total_generated[label] = int(approx_total_gen * sc_dist.get(sc, 0)) # Approximate generated
            total_transmitted[label] = df[f"tx_pkts_{label}"].sum()
            # Dropped is cumulative, take the last value
            total_dropped[label] = df[f"dropped_pkts_{label}"].iloc[-1] if not df.empty else 0

            # Calculate latency stats from transmitted packets
            latencies = [p.latency for p in self.all_transmitted_packets if p.service_class == sc and p.departure_time >= 0]
            total_latency[label] = sum(latencies)
            packet_counts[label] = len(latencies)
            avg_latency[label] = (total_latency[label] / packet_counts[label]) if packet_counts[label] > 0 else 0


        summary["generated_approx"] = total_generated
        summary["transmitted"] = total_transmitted
        summary["dropped"] = total_dropped
        summary["avg_latency_ms"] = {k: v * 1000 for k, v in avg_latency.items()} # Convert to ms
        summary["avg_bw_utilization_percent"] = df["bw_utilization_percent"].mean() if not df.empty else 0

        return summary