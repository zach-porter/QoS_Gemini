# sim/queues.py
from collections import deque
from typing import Dict, List, Optional
from .packet import Packet
from .service_class import ServiceClass, PRIORITY_ORDER

class QoSQueues:
    """Manages separate queues for each service class."""

    def __init__(self, max_queue_size_packets: Optional[int] = None):
        """
        Initializes QoS queues.

        Args:
            max_queue_size_packets: Optional limit on packets per queue.
                                     If None, queues are unbounded.
        """
        self._queues: Dict[ServiceClass, deque] = {sc: deque() for sc in ServiceClass}
        self._queue_byte_sizes: Dict[ServiceClass, int] = {sc: 0 for sc in ServiceClass}
        self.max_queue_size_packets = max_queue_size_packets
        self._dropped_packets: Dict[ServiceClass, int] = {sc: 0 for sc in ServiceClass}

    def enqueue(self, packet: Packet) -> bool:
        """
        Adds a packet to the appropriate queue.

        Args:
            packet: The packet to enqueue.

        Returns:
            True if the packet was successfully enqueued, False if dropped.
        """
        queue = self._queues[packet.service_class]
        if self.max_queue_size_packets is not None and len(queue) >= self.max_queue_size_packets:
            self._dropped_packets[packet.service_class] += 1
            # print(f"Dropped packet {packet.packet_id} from {packet.service_class.label} queue (Max size reached)")
            return False
        else:
            queue.append(packet)
            self._queue_byte_sizes[packet.service_class] += packet.size_bytes
            return True

    def dequeue(self, service_class: ServiceClass) -> Optional[Packet]:
        """Removes and returns the next packet from the specified queue."""
        queue = self._queues[service_class]
        if queue:
            packet = queue.popleft()
            self._queue_byte_sizes[service_class] -= packet.size_bytes
            return packet
        return None

    def peek(self, service_class: ServiceClass) -> Optional[Packet]:
        """Returns the next packet without removing it."""
        queue = self._queues[service_class]
        if queue:
            return queue[0]
        return None

    def get_queue_packet_count(self, service_class: ServiceClass) -> int:
        """Returns the number of packets in a specific queue."""
        return len(self._queues[service_class])

    def get_queue_byte_size(self, service_class: ServiceClass) -> int:
        """Returns the total byte size of packets in a specific queue."""
        # Recalculate for safety, though _queue_byte_sizes should be accurate
        # return sum(p.size_bytes for p in self._queues[service_class])
        return self._queue_byte_sizes[service_class]

    def get_total_queued_bytes(self) -> int:
        """Returns the total byte size across all queues."""
        return sum(self._queue_byte_sizes.values())

    def get_dropped_count(self, service_class: ServiceClass) -> int:
        """Returns the number of dropped packets for a specific class."""
        return self._dropped_packets[service_class]

    def get_all_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """Returns packet counts and byte sizes for all queues."""
        stats = {}
        for sc in PRIORITY_ORDER:
            stats[sc.label] = {
                "packets": self.get_queue_packet_count(sc),
                "bytes": self.get_queue_byte_size(sc),
                "dropped": self.get_dropped_count(sc)
            }
        return stats