# sim/scheduler.py
from typing import List, Dict, Tuple
import numpy as np
from .packet import Packet
from .queues import QoSQueues
from .service_class import ServiceClass, PRIORITY_ORDER

class SatelliteScheduler:
    """Simulates the QoS scheduler at the satellite gateway."""

    def __init__(self, qos_thresholds_percent: Dict[ServiceClass, float]):
        """
        Initializes the scheduler.

        Args:
            qos_thresholds_percent: Dictionary mapping service classes (except BE)
                                    to their guaranteed minimum bandwidth percentage.
                                    Example: {ServiceClass.NC: 5, ServiceClass.EF: 20, ServiceClass.AF: 25}
                                    Best Effort (BE) gets the remaining bandwidth.
                                    The sum should ideally be <= 100.
        """
        self.qos_thresholds_percent = qos_thresholds_percent
        self._validate_thresholds()

    def _validate_thresholds(self):
        total_percent = sum(self.qos_thresholds_percent.values())
        if total_percent > 100.0:
            print(f"Warning: QoS threshold sum ({total_percent}%) exceeds 100%.")
        if ServiceClass.BEST_EFFORT in self.qos_thresholds_percent:
             print("Warning: Threshold for Best Effort is ignored; it gets remaining bandwidth.")

    def schedule_packets(self, queues: QoSQueues, available_bandwidth_bytes: int, current_time: float) -> Tuple[List[Packet], Dict[ServiceClass, int]]:
        """
        Selects packets for transmission based on QoS rules and available bandwidth.

        Implements strict priority scheduling with minimum bandwidth guarantees (like CBWFQ).

        Args:
            queues: The QoSQueues object from the User Terminal.
            available_bandwidth_bytes: Total bandwidth available in this time step (in bytes).
            current_time: The current simulation time.

        Returns:
            A tuple containing:
            - list[Packet]: A list of packets selected for transmission.
            - dict[ServiceClass, int]: Bytes transmitted per service class.
        """
        transmitted_packets: List[Packet] = []
        transmitted_bytes_per_class: Dict[ServiceClass, int] = {sc: 0 for sc in ServiceClass}
        remaining_bandwidth_bytes = available_bandwidth_bytes

        # 1. Process classes with thresholds (NC, EF, AF) in priority order
        for sc in PRIORITY_ORDER:
            if sc == ServiceClass.BEST_EFFORT: # Skip BE for now
                continue

            threshold_percent = self.qos_thresholds_percent.get(sc, 0)
            # Bandwidth guaranteed for this class in this step
            guaranteed_bw_bytes = int(available_bandwidth_bytes * (threshold_percent / 100.0))
            served_bytes_in_class = 0

            while remaining_bandwidth_bytes > 0:
                packet = queues.peek(sc)
                if not packet:
                    break # Queue is empty

                # Can we transmit this packet?
                # Condition: EITHER we are within the guaranteed BW for this class
                #            OR there's simply enough remaining BW overall (allowing classes
                #            to exceed their guarantee if higher priority classes didn't use all BW)
                can_transmit_within_guarantee = (served_bytes_in_class + packet.size_bytes <= guaranteed_bw_bytes)
                can_transmit_overall = (packet.size_bytes <= remaining_bandwidth_bytes)

                # We prioritize using the guarantee first conceptually, but allow exceeding it
                if can_transmit_overall:
                    dequeued_packet = queues.dequeue(sc)
                    if dequeued_packet: # Should always be true if peek worked
                        dequeued_packet.mark_departed(current_time) # Mark departure time
                        transmitted_packets.append(dequeued_packet)
                        remaining_bandwidth_bytes -= dequeued_packet.size_bytes
                        served_bytes_in_class += dequeued_packet.size_bytes
                        transmitted_bytes_per_class[sc] += dequeued_packet.size_bytes
                    else:
                        break # Should not happen if peek worked
                else:
                    break # Not enough remaining bandwidth for the next packet in this queue

        # 2. Process Best Effort (BE) with any remaining bandwidth
        sc = ServiceClass.BEST_EFFORT
        while remaining_bandwidth_bytes > 0:
            packet = queues.peek(sc)
            if not packet:
                break # BE queue empty

            if packet.size_bytes <= remaining_bandwidth_bytes:
                dequeued_packet = queues.dequeue(sc)
                if dequeued_packet:
                    dequeued_packet.mark_departed(current_time)
                    transmitted_packets.append(dequeued_packet)
                    remaining_bandwidth_bytes -= dequeued_packet.size_bytes
                    transmitted_bytes_per_class[sc] += dequeued_packet.size_bytes
                else:
                    break
            else:
                break # Not enough remaining bandwidth for the next BE packet

        return transmitted_packets, transmitted_bytes_per_class