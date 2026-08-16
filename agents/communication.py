"""Multi-agent communication protocols for supply-chain coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class CommTopology(str, Enum):
    NONE = "none"
    PAIRWISE = "pairwise"          # only adjacent echelons
    NEIGHBOR = "neighbor"          # same as pairwise for linear chain
    FULL = "full"                  # all-to-all
    CENTRAL = "central"            # all agents talk to a coordinator
    NEGOTIATION = "negotiation"    # structured offer/counter-offer style


@dataclass
class Message:
    sender: str
    receiver: str                  # or "broadcast" / "coordinator"
    content: Dict[str, Any]
    period: int
    msg_type: str = "info"         # info | request | offer | ack

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "period": self.period,
            "msg_type": self.msg_type,
        }


class CommunicationBus:
    """
    Lightweight message bus. Agents never mutate the environment via messages;
    they only exchange structured information that can be included in observations.
    """

    def __init__(self, topology: CommTopology | str = CommTopology.NONE, echelon_order: Optional[List[str]] = None):
        self.topology = CommTopology(topology) if isinstance(topology, str) else topology
        self.echelon_order = echelon_order or ["retailer", "wholesaler", "distributor", "manufacturer"]
        self.inbox: Dict[str, List[Message]] = {e: [] for e in self.echelon_order}
        self.inbox["coordinator"] = []
        self.history: List[Message] = []
        self.message_count = 0

    def reset(self) -> None:
        for k in self.inbox:
            self.inbox[k] = []
        self.history.clear()
        self.message_count = 0

    def allowed_receivers(self, sender: str) -> List[str]:
        if self.topology == CommTopology.NONE:
            return []
        if self.topology in (CommTopology.PAIRWISE, CommTopology.NEIGHBOR):
            if sender not in self.echelon_order:
                return []
            idx = self.echelon_order.index(sender)
            recv = []
            if idx > 0:
                recv.append(self.echelon_order[idx - 1])
            if idx < len(self.echelon_order) - 1:
                recv.append(self.echelon_order[idx + 1])
            return recv
        if self.topology == CommTopology.FULL:
            return [e for e in self.echelon_order if e != sender]
        if self.topology in (CommTopology.CENTRAL, CommTopology.NEGOTIATION):
            if sender == "coordinator":
                return list(self.echelon_order)
            return ["coordinator"]
        return []

    def send(self, msg: Message) -> bool:
        """Deliver a message if topology permits. Returns True if delivered."""
        allowed = self.allowed_receivers(msg.sender)
        if msg.receiver == "broadcast":
            delivered = False
            for r in allowed:
                m = Message(sender=msg.sender, receiver=r, content=msg.content,
                            period=msg.period, msg_type=msg.msg_type)
                self.inbox.setdefault(r, []).append(m)
                self.history.append(m)
                self.message_count += 1
                delivered = True
            return delivered
        if msg.receiver not in allowed and msg.receiver != "coordinator":
            # still allow explicit coordinator if topology is central
            if not (self.topology in (CommTopology.CENTRAL, CommTopology.NEGOTIATION) and msg.receiver == "coordinator"):
                return False
        self.inbox.setdefault(msg.receiver, []).append(msg)
        self.history.append(msg)
        self.message_count += 1
        return True

    def receive(self, agent_name: str) -> List[Message]:
        msgs = list(self.inbox.get(agent_name, []))
        self.inbox[agent_name] = []
        return msgs

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": self.message_count,
            "topology": self.topology.value,
        }
