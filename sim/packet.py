# sim/packet.py
import itertools
from .service_class import ServiceClass

class Packet:
    """Represents a data packet."""
    _ids = itertools.count(1) # Class-level counter for unique IDs

    def __init__(self, service_class: ServiceClass, size_bytes: int, arrival_time: float):
        """
        Initializes a Packet.

        Args:
            service_class: The QoS class of the packet.
            size_bytes: The size of the packet in bytes.
            arrival_time: The simulation time when the packet arrived at the queue.
        """
        self.packet_id = next(Packet._ids)
        self.service_class = service_class
        self.size_bytes = size_bytes
        self.arrival_time = arrival_time
        self.departure_time = -1.0 # Mark as not departed

    def __repr__(self):
        return (f"Packet(ID={self.packet_id}, Class={self.service_class.label}, "
                f"Size={self.size_bytes}B, Arrival={self.arrival_time:.2f})")

    def mark_departed(self, departure_time: float):
        self.departure_time = departure_time

    @property
    def latency(self) -> float:
        """Calculates packet latency if it has departed."""
        if self.departure_time >= 0:
            return self.departure_time - self.arrival_time
        return float('inf') # Not departed yet