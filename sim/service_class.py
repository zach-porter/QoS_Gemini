# sim/service_class.py
from enum import Enum

class ServiceClass(Enum):
    """Enumeration for different Quality of Service classes."""
    NETWORK_CONTROL = ("NC", 1)      # Highest priority
    EXPEDITED_FORWARDING = ("EF", 2)
    ASSURED_FORWARDING = ("AF", 3)
    BEST_EFFORT = ("BE", 4)          # Lowest priority

    def __init__(self, label, priority):
        self.label = label
        self.priority = priority

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.priority < other.priority
        return NotImplemented

# Define priority order for easy iteration
PRIORITY_ORDER = sorted(list(ServiceClass), key=lambda sc: sc.priority)

# Example usage:
# print(ServiceClass.NETWORK_CONTROL.label)  # Output: NC
# print(ServiceClass.EXPEDITED_FORWARDING.priority) # Output: 2
# print(PRIORITY_ORDER) # Output: List sorted by priority