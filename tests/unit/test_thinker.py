# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T5.2: ThinkerList, EventBus, and GameSim."""

from __future__ import annotations

from typing import Any

from pyhexen.core.events import EventBus, MapExit, PrintMessage, SectorChanged, SoundEvent
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.thinker import Thinker, ThinkerList


class RecordingThinker(Thinker):
    """Test thinker that records its name into a log on each tick."""

    def __init__(self, name: str, log: list[str], on_tick: Any = None) -> None:
        super().__init__()
        self.name = name
        self.log = log
        self.on_tick = on_tick
        self.ticks = 0

    def tick(self, sim: GameSim) -> None:
        self.ticks += 1
        self.log.append(self.name)
        if self.on_tick is not None:
            self.on_tick(self, sim)

    def state_tuple(self) -> tuple[Any, ...]:
        return ("RecordingThinker", self.name, self.ticks)


def test_thinker_list_order_preservation() -> None:
    sim = GameSim()
    log: list[str] = []
    t1 = RecordingThinker("A", log)
    t2 = RecordingThinker("B", log)
    t3 = RecordingThinker("C", log)

    sim.thinkers.add(t1)
    sim.thinkers.add(t2)
    sim.thinkers.add(t3)

    assert len(sim.thinkers) == 3
    sim.tick()
    assert log == ["A", "B", "C"]
    sim.tick()
    assert log == ["A", "B", "C", "A", "B", "C"]


def test_thinker_remove_self_during_iteration() -> None:
    sim = GameSim()
    log: list[str] = []

    def remove_self(t: RecordingThinker, s: GameSim) -> None:
        s.thinkers.remove(t)

    t1 = RecordingThinker("A", log)
    t2 = RecordingThinker("B", log, on_tick=remove_self)
    t3 = RecordingThinker("C", log)

    sim.thinkers.add(t1)
    sim.thinkers.add(t2)
    sim.thinkers.add(t3)

    sim.tick()
    # B ran once and marked itself removed; A and C also ran
    assert log == ["A", "B", "C"]
    assert len(sim.thinkers) == 2

    log.clear()
    sim.tick()
    # On the next tic, B is unlinked and skipped
    assert log == ["A", "C"]


def test_thinker_remove_next_and_previous_during_iteration() -> None:
    sim = GameSim()
    log: list[str] = []

    t1 = RecordingThinker("A", log)
    t2 = RecordingThinker("B", log)
    t3 = RecordingThinker("C", log)
    t4 = RecordingThinker("D", log)

    # When B ticks, it removes both A (previous) and C (next)
    def remove_neighbors(_t: RecordingThinker, s: GameSim) -> None:
        s.thinkers.remove(t1)
        s.thinkers.remove(t3)

    t2.on_tick = remove_neighbors

    sim.thinkers.add(t1)
    sim.thinkers.add(t2)
    sim.thinkers.add(t3)
    sim.thinkers.add(t4)

    sim.tick()
    # A ran before B; B marked A & C removed; C is unlinked without ticking; D runs
    assert log == ["A", "B", "D"]
    assert len(sim.thinkers) == 2

    log.clear()
    t2.on_tick = None
    sim.tick()
    assert log == ["B", "D"]


def test_thinker_add_during_iteration() -> None:
    sim = GameSim()
    log: list[str] = []

    def spawn_new(t: RecordingThinker, s: GameSim) -> None:
        if t.ticks == 1:
            s.thinkers.add(RecordingThinker("NEW", log))

    t1 = RecordingThinker("A", log, on_tick=spawn_new)
    t2 = RecordingThinker("B", log)
    sim.thinkers.add(t1)
    sim.thinkers.add(t2)

    sim.tick()
    # In p_tick.c, a thinker added at the tail during RunThinkers runs at the end of the same tic
    assert log == ["A", "B", "NEW"]


def test_thinker_init_resets_list() -> None:
    tl = ThinkerList()
    log: list[str] = []
    tl.add(RecordingThinker("A", log))
    assert len(tl) == 1
    tl.init_thinkers()
    assert len(tl) == 0


def test_event_bus() -> None:
    bus = EventBus()
    received_sectors: list[SectorChanged] = []
    bus.subscribe(SectorChanged, received_sectors.append)

    e1 = SectorChanged(sector_index=5)
    e2 = SoundEvent(sound_id=10, x=100, y=200)
    e3 = PrintMessage(text="GREETINGS MORTAL", bold=True)
    e4 = MapExit(next_map=2, position=1)

    bus.emit(e1)
    bus.emit(e2)
    bus.emit(e3)
    bus.emit(e4)

    assert received_sectors == [e1]
    drained = bus.drain()
    assert drained == [e1, e2, e3, e4]
    assert bus.drain() == []

    bus.emit(e1)
    bus.clear()
    assert bus.drain() == []


def test_game_sim_tick_and_state_hash() -> None:
    sim1 = GameSim()
    sim2 = GameSim()

    assert sim1.state_hash() == sim2.state_hash()

    log1: list[str] = []
    log2: list[str] = []

    def rng_step(_t: RecordingThinker, s: GameSim) -> None:
        s.rng.p_random()

    sim1.thinkers.add(RecordingThinker("T1", log1, on_tick=rng_step))
    sim2.thinkers.add(RecordingThinker("T1", log2, on_tick=rng_step))

    for _ in range(10):
        sim1.tick()
        sim2.tick()

    assert sim1.level_time == 10
    assert sim1.state_hash() == sim2.state_hash()

    # Advancing sim1 one more tic changes the hash
    sim1.tick()
    assert sim1.state_hash() != sim2.state_hash()

    # Paused sim does not advance level_time or thinkers
    hash_before_pause = sim1.state_hash()
    sim1.paused = True
    sim1.tick()
    assert sim1.level_time == 11
    assert sim1.state_hash() == hash_before_pause
