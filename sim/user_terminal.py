# sim/user_terminal.py
import random
import numpy as np
from .packet import Packet
from .queues import QoSQueues
from .service_class import ServiceClass, PRIORITY_ORDER

class UserTerminal:
    """Simulates a user terminal generating packets."""

    def __init__(self, terminal_id: int, config: dict):
        """
        Initializes a User Terminal.

        Args:
            terminal_id: A unique ID for the terminal.
            config: Dictionary containing configuration like packet generation rates,
                    size distributions, service class mix, max queue size.
                    Example:
                    {
                        'packet_rate_pps': 10, # Avg packets per second
                        'packet_size_bytes_avg': 1000,
                        'packet_size_bytes_stddev': 200,
                        'service_class_distribution': {
                            ServiceClass.NETWORK_CONTROL: 0.05,
                            ServiceClass.EXPEDITED_FORWARDING: 0.15,
                            ServiceClass.ASSURED_FORWARDING: 0.30,
                            ServiceClass.BEST_EFFORT: 0.50
                        },
                        'max_queue_size_packets': 1000 # Optional
                    }
        """
        self.terminal_id = terminal_id
        self.config = config
        self.queues = QoSQueues(max_queue_size_packets=config.get('max_queue_size_packets'))
        self._packet_id_counter = 0

        # Validate distribution sums to 1
        dist_sum = sum(config['service_class_distribution'].values())
        if not np.isclose(dist_sum, 1.0):
             raise ValueError("Service class distribution must sum to 1.0")

        self._sc_list = list(config['service_class_distribution'].keys())
        self._sc_weights = list(config['service_class_distribution'].values())

        # For Poisson arrival process
        self._avg_interarrival_time = 1.0 / config['packet_rate_pps'] if config['packet_rate_pps'] > 0 else float('inf')
        self._next_arrival_time = 0 # Initialize

    def _generate_next_arrival_time(self, current_time: float) -> float:
         # Generate next arrival time using exponential distribution (Poisson process)
         return current_time + random.expovariate(1.0 / self._avg_interarrival_time)

    def _generate_packet_size(self) -> int:
        # Generate packet size using a normal distribution, ensure positive
        size = int(random.normalvariate(
            self.config['packet_size_bytes_avg'],
            self.config['packet_size_bytes_stddev']
        ))
        return max(50, size) # Ensure a minimum packet size (e.g., header size)

    def _choose_service_class(self) -> ServiceClass:
        # Choose service class based on the defined distribution
        return random.choices(self._sc_list, weights=self._sc_weights, k=1)[0]

    def generate_packets(self, current_time: float, time_step: float):
        """
        Generates packets based on arrival rate and enqueues them.
        Uses a Poisson process for packet arrivals within the time step.
        """
        # Initialize next arrival if it's the first run or needs reset
        if self._next_arrival_time < current_time:
             self._next_arrival_time = self._generate_next_arrival_time(current_time)

        # Generate packets that should arrive within this time step
        while self._next_arrival_time < current_time + time_step:
            arrival_time = self._next_arrival_time
            service_class = self._choose_service_class()
            size_bytes = self._generate_packet_size()

            packet = Packet(
                service_class=service_class,
                size_bytes=size_bytes,
                arrival_time=arrival_time
            )
            self.queues.enqueue(packet)
            # print(f"Time {arrival_time:.2f}: UT {self.terminal_id} generated {packet}") # DEBUG

            # Schedule the next arrival
            self._next_arrival_time = self._generate_next_arrival_time(arrival_time)

    def get_queues(self) -> QoSQueues:
        """Returns the internal QoS queues object."""
        return self.queues