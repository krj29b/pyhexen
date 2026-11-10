# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for packaging, first-run IWAD dialog, crash logs, and PyInstaller spec (T14.4)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.app.config import load_config
from pyhexen.app.crash import (
    crash_guard,
    format_crash_report,
    write_crash_log,
)
from pyhexen.app.dialog import (
    prompt_for_iwad,
    show_error_dialog,
    validate_hexen_iwad,
)


def test_validate_hexen_iwad_valid(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD")
    wb.add_playpal(1).add_colormap(32)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=128, height=128)
    wb.add_map(mb)
    wb.add_lump("MAPINFO", b'map MAP01 "Test"\n')
    valid_wad = tmp_path / "hexen.wad"
    valid_wad.write_bytes(wb.to_bytes())

    valid, reason = validate_hexen_iwad(valid_wad)
    assert valid is True
    assert "Valid" in reason


def test_validate_hexen_iwad_missing_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "nope.wad"
    valid, reason = validate_hexen_iwad(non_existent)
    assert valid is False
    assert "does not exist" in reason


def test_validate_hexen_iwad_invalid_header(tmp_path: Path) -> None:
    bad_wad = tmp_path / "corrupt.wad"
    bad_wad.write_bytes(b"NOTAWADHEADER")
    valid, reason = validate_hexen_iwad(bad_wad)
    assert valid is False
    assert "Invalid WAD magic" in reason


def test_validate_hexen_iwad_missing_lumps(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD")
    wb.add_lump("SOME_LUMP", b"123456")
    incomplete_wad = tmp_path / "incomplete.wad"
    incomplete_wad.write_bytes(wb.to_bytes())

    valid, reason = validate_hexen_iwad(incomplete_wad)
    assert valid is False
    assert "MAP01" in reason or "PLAYPAL" in reason or "MAPINFO" in reason


def test_prompt_for_iwad_success(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD")
    wb.add_playpal(1).add_colormap(32)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=128, height=128)
    wb.add_map(mb)
    wb.add_lump("MAPINFO", b'map MAP01 "Test"\n')
    valid_wad = tmp_path / "hexen.wad"
    valid_wad.write_bytes(wb.to_bytes())

    cfg_file = tmp_path / "config.toml"
    with patch("pyhexen.app.dialog.prompt_for_iwad") as mock_prompt:
        mock_prompt.return_value = valid_wad
        res = mock_prompt()
        assert res == valid_wad

    with patch("tkinter.filedialog.askopenfilename", return_value=str(valid_wad)):
        with patch("tkinter.Tk"):
            found = prompt_for_iwad(config_path=cfg_file, use_gui=True)
            assert found == valid_wad.resolve()
            saved_cfg = load_config(cfg_file)
            assert Path(saved_cfg.iwad).resolve() == valid_wad.resolve()


def test_prompt_for_iwad_cancel(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    with patch("tkinter.filedialog.askopenfilename", return_value=""):
        with patch("tkinter.Tk"):
            found = prompt_for_iwad(config_path=cfg_file, use_gui=True)
            assert found is None


def test_prompt_for_iwad_gui_failure_fallback(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    with patch("tkinter.Tk", side_effect=Exception("No display")):
        found = prompt_for_iwad(config_path=cfg_file, use_gui=True)
        assert found is None


def test_show_error_dialog_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("tkinter.Tk", side_effect=Exception("No display")):
        show_error_dialog("Test Error", "Something went wrong")
    captured = capsys.readouterr()
    assert "[Test Error]" in captured.err
    assert "Something went wrong" in captured.err


def test_format_crash_report() -> None:
    try:
        raise ValueError("Simulated engine explosion")
    except ValueError as e:
        report = format_crash_report(e, context_info={"map": "MAP01", "tics": 350})

    assert "PyHexen Fatal Crash Report" in report
    assert "ValueError: Simulated engine explosion" in report
    assert "map: MAP01" in report
    assert "tics: 350" in report
    assert "Python Version:" in report
    assert "OS / Platform:" in report


def test_write_crash_log(tmp_path: Path) -> None:
    log_path = tmp_path / "crash.log"
    try:
        raise RuntimeError("Crash log test")
    except RuntimeError as e:
        written = write_crash_log(e, context_info={"skill": 4}, log_path=log_path)

    assert written == log_path
    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "RuntimeError: Crash log test" in content
    assert "skill: 4" in content


def test_crash_guard_catches_and_logs(tmp_path: Path) -> None:
    log_path = tmp_path / "test_crash.log"
    with pytest.raises(SystemExit) as exc_info:
        with crash_guard(context_info={"phase": "test"}, log_path=log_path, show_dialog=False):
            raise ArithmeticError("Division by zero in test")

    assert exc_info.value.code == 1
    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "ArithmeticError: Division by zero in test" in content
    assert "phase: test" in content


def test_crash_guard_allows_system_exit(tmp_path: Path) -> None:
    log_path = tmp_path / "exit.log"
    with pytest.raises(SystemExit) as exc_info:
        with crash_guard(log_path=log_path, show_dialog=False):
            sys.exit(0)
    assert exc_info.value.code == 0
    assert not log_path.exists()


def test_pyinstaller_spec_structure() -> None:
    spec_path = Path("pyhexen.spec")
    assert spec_path.is_file(), "pyhexen.spec must exist in project root"
    spec_text = spec_path.read_text(encoding="utf-8")

    # Verify shader datas are bundled
    assert "*.vert" in spec_text
    assert "*.frag" in spec_text
    assert "pyhexen/render/shaders" in spec_text

    # Verify strict game data exclusions
    assert "*.wad" in spec_text
    assert "*.WAD" in spec_text

    # Verify entry point
    assert "src/pyhexen/__main__.py" in spec_text or "__main__.py" in spec_text
    # Verify console/onedir collection
    assert "COLLECT" in spec_text
    assert "EXE" in spec_text
