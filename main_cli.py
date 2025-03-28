# main_cli.py
import pandas as pd
from sim.simulation import Simulation
from sim.config import get_default_config

if __name__ == "__main__":
    # Load default configuration
    config = get_default_config()

    # --- Optional: Modify config here if needed ---
    # config["simulation_duration_sec"] = 10.0
    # config["ut_config"]["packet_rate_pps"] = 50
    # config["bandwidth_type"] = "dynamic"
    # ---------------------------------------------

    # Create and run the simulation
    sim = Simulation(config)
    sim.run_simulation()

    # Get and display results
    results_df = sim.get_results_dataframe()
    summary_stats = sim.get_summary_stats()

    print("\n--- Simulation Results ---")
    if not results_df.empty:
        print("Last few time steps:")
        print(results_df.tail())
    else:
        print("No simulation history recorded.")

    print("\n--- Summary Statistics ---")
    import json
    print(json.dumps(summary_stats, indent=2))

    # You can also save the full history to CSV
    # results_df.to_csv("simulation_results.csv", index=False)
    # print("\nFull results saved to simulation_results.csv")