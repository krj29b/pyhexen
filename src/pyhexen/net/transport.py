# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Datagram transports: UDP sockets, and an in-process loopback with delay and loss.

Both are non-blocking: `recv()` returns whatever has arrived. The loopback network
takes a clock function, so tests can run with virtual time.
"""

from __future__ import annotations

import errno
import heapq
import socket
from collections.abc import Callable
from typing import Protocol

Address = tuple[str, int]
MAX_DATAGRAM = 4096


class Transport(Protocol):
    """What the server and client need from a socket."""

    address: Address

    def send(self, addr: Address, data: bytes) -> None: ...

    def recv(self) -> list[tuple[Address, bytes]]: ...

    def close(self) -> None: ...


class UdpTransport:
    """A non-blocking UDP socket (IPv4)."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.bind((host, port))
        bound = self._sock.getsockname()
        self.address: Address = (bound[0], bound[1])

    def send(self, addr: Address, data: bytes) -> None:
        try:
            self._sock.sendto(data, addr)
        except OSError:
            pass  # unreachable or full: the next packet repeats everything

    def recv(self) -> list[tuple[Address, bytes]]:
        out: list[tuple[Address, bytes]] = []
        while True:
            try:
                data, addr = self._sock.recvfrom(MAX_DATAGRAM)
            except BlockingIOError:
                break
            except ConnectionResetError:
                # Windows reports an ICMP "port unreachable" from an earlier send here.
                continue
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                continue
            out.append(((addr[0], addr[1]), data))
        return out

    def close(self) -> None:
        self._sock.close()


class _Xorshift:
    def __init__(self, seed: int) -> None:
        self.state = (seed * 2654435761) & 0xFFFFFFFF or 1

    def next_float(self) -> float:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x / 0x100000000


class LoopbackNetwork:
    """In-process datagram network with a fixed one-way delay, jitter and loss."""

    def __init__(
        self,
        clock: Callable[[], float],
        *,
        delay: float = 0.0,
        jitter: float = 0.0,
        loss: float = 0.0,
        seed: int = 1,
    ) -> None:
        self.clock = clock
        self.delay = delay
        self.jitter = jitter
        self.loss = loss
        self._rng = _Xorshift(seed)
        self._queue: list[tuple[float, int, Address, Address, bytes]] = []
        self._seq = 0
        self._endpoints: dict[Address, LoopbackTransport] = {}
        self.sent = 0
        self.dropped = 0

    def endpoint(self, addr: Address) -> LoopbackTransport:
        ep = LoopbackTransport(self, addr)
        self._endpoints[addr] = ep
        return ep

    def post(self, src: Address, dst: Address, data: bytes) -> None:
        self.sent += 1
        if self.loss and self._rng.next_float() < self.loss:
            self.dropped += 1
            return
        when = self.clock() + self.delay + self.jitter * self._rng.next_float()
        self._seq += 1
        heapq.heappush(self._queue, (when, self._seq, dst, src, data))

    def deliver(self) -> None:
        now = self.clock()
        while self._queue and self._queue[0][0] <= now:
            _, _, dst, src, data = heapq.heappop(self._queue)
            ep = self._endpoints.get(dst)
            if ep is not None and not ep.closed:
                ep.inbox.append((src, data))


class LoopbackTransport:
    """One endpoint of a `LoopbackNetwork`."""

    def __init__(self, network: LoopbackNetwork, addr: Address) -> None:
        self.network = network
        self.address: Address = addr
        self.inbox: list[tuple[Address, bytes]] = []
        self.closed = False

    def send(self, addr: Address, data: bytes) -> None:
        if not self.closed:
            self.network.post(self.address, addr, data)

    def recv(self) -> list[tuple[Address, bytes]]:
        self.network.deliver()
        out, self.inbox = self.inbox, []
        return out

    def close(self) -> None:
        self.closed = True
