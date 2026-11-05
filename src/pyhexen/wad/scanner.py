# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Script text tokenizer for Hexen text lumps (MAPINFO, SNDINFO, SNDSEQ, ANIMDEFS).

Port of sc_man.c from Chocolate Doom (src/hexen/sc_man.c).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from pyhexen.wad.wadfile import WadFile, WadStack

ASCII_COMMENT = ";"
ASCII_QUOTE = '"'


class ScriptError(ValueError):
    """Raised when a script lump contains a syntax or token error."""


class Scanner:
    """Tokenizer for Hexen script lumps.

    Port of sc_man.c (SC_Open, SC_GetString, SC_MustGetString, SC_GetNumber,
    SC_MustGetNumber, SC_UnGet, SC_Compare, SC_MatchString, SC_MustMatchString).
    """

    def __init__(self, text: str | bytes, *, name: str = "<script>") -> None:
        if isinstance(text, bytes):
            # Strip trailing NUL bytes if present in lump data
            decoded = text.split(b"\x00", 1)[0].decode("latin1")
        else:
            decoded = text.split("\x00", 1)[0]

        self._text: str = decoded
        self._len: int = len(decoded)
        self._pos: int = 0
        self._already_got: bool = False

        self.name: str = name
        self.string: str = ""
        self.number: int = 0
        self.line: int = 1
        self.end: bool = False
        self.crossed: bool = False
        self.quoted: bool = False

    @classmethod
    def from_wad(cls, wad: WadStack | WadFile, lump_name: str) -> Scanner:
        """Open a script lump from a WAD or WadStack."""
        raw = wad.get_bytes(lump_name)
        return cls(raw, name=lump_name.upper())

    def get_string(self) -> bool:
        """Fetch the next token into `self.string`. Returns False at EOF.

        Port of SC_GetString (sc_man.c).
        """
        if self._already_got:
            self._already_got = False
            return True

        self.crossed = False
        self.quoted = False
        text = self._text
        n = self._len

        if self._pos >= n:
            self.end = True
            return False

        # Skip whitespace and comments (; line comments, // line comments, /* */ block comments)
        while True:
            while self._pos < n and ord(text[self._pos]) <= 32:
                if text[self._pos] == "\n":
                    self.line += 1
                    self.crossed = True
                self._pos += 1

            if self._pos >= n:
                self.end = True
                return False

            ch = text[self._pos]
            if ch == ASCII_COMMENT or text.startswith("//", self._pos):
                while self._pos < n and text[self._pos] != "\n":
                    self._pos += 1
                if self._pos >= n:
                    self.end = True
                    return False
                # Consume the newline terminating the comment
                self._pos += 1
                self.line += 1
                self.crossed = True
            elif text.startswith("/*", self._pos):
                self._pos += 2
                while self._pos < n and not text.startswith("*/", self._pos):
                    if text[self._pos] == "\n":
                        self.line += 1
                        self.crossed = True
                    self._pos += 1
                if self._pos >= n:
                    self.end = True
                    return False
                self._pos += 2
            else:
                break

        if text[self._pos] == ASCII_QUOTE:
            self.quoted = True
            self._pos += 1
            start = self._pos
            while self._pos < n and text[self._pos] != ASCII_QUOTE:
                if text[self._pos] == "\n":
                    self.line += 1
                    self.crossed = True
                self._pos += 1
            self.string = text[start : self._pos]
            if self._pos < n and text[self._pos] == ASCII_QUOTE:
                self._pos += 1
        else:
            start = self._pos
            while self._pos < n:
                c = text[self._pos]
                if ord(c) <= 32 or c == ASCII_COMMENT or c == ASCII_QUOTE:
                    break
                if text.startswith("//", self._pos) or text.startswith("/*", self._pos):
                    break
                self._pos += 1
            self.string = text[start : self._pos]

        return True

    def must_get_string(self) -> str:
        """Fetch the next token or raise ScriptError if at EOF.

        Port of SC_MustGetString (sc_man.c).
        """
        if not self.get_string():
            self.script_error("Missing string.")
        return self.string

    def must_get_string_name(self, expected: str) -> None:
        """Fetch the next token and verify it matches `expected` (case-insensitive).

        Port of SC_MustGetStringName (sc_man.c).
        """
        self.must_get_string()
        if not self.compare(expected):
            self.script_error(f"Expected {expected!r}, got {self.string!r}.")

    def get_number(self) -> bool:
        """Fetch the next token and parse it as an integer into `self.number`.

        Port of SC_GetNumber (sc_man.c).
        """
        if self.get_string():
            try:
                self.number = int(self.string, 0)
            except ValueError:
                raise ScriptError(
                    f'SC_GetNumber: Bad numeric constant "{self.string}".\n'
                    f"Script {self.name}, Line {self.line}"
                ) from None
            return True
        return False

    def must_get_number(self) -> int:
        """Fetch the next integer token or raise ScriptError if missing or invalid.

        Port of SC_MustGetNumber (sc_man.c).
        """
        if not self.get_number():
            self.script_error("Missing integer.")
        return self.number

    def unget(self) -> None:
        """Push the current token back so the next `get_string()` returns it again.

        Port of SC_UnGet (sc_man.c).
        """
        self._already_got = True

    def compare(self, expected: str) -> bool:
        """Case-insensitive comparison between `self.string` and `expected`.

        Port of SC_Compare (sc_man.c).
        """
        return self.string.lower() == expected.lower()

    def match_string(self, strings: Sequence[str]) -> int:
        """Return index of first case-insensitive match in `strings`, or -1.

        Port of SC_MatchString (sc_man.c).
        """
        for i, candidate in enumerate(strings):
            if self.compare(candidate):
                return i
        return -1

    def must_match_string(self, strings: Sequence[str]) -> int:
        """Return index of first match in `strings`, or raise ScriptError.

        Port of SC_MustMatchString (sc_man.c).
        """
        idx = self.match_string(strings)
        if idx == -1:
            self.script_error(None)
        return idx

    def script_error(self, message: str | None = None) -> NoReturn:
        """Raise a ScriptError formatted with the script name and current line.

        Port of SC_ScriptError (sc_man.c).
        """
        msg = message if message is not None else "Bad syntax."
        raise ScriptError(f'Script error, "{self.name}" line {self.line}: {msg}')
