"""Readable atom-selection expressions for CrossRMSD."""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass

import numpy as np

from crossrmsd.models import AtomRecord

_SELECTOR_NAMES = {"name", "resname", "resid", "chain", "element", "index"}


def tokenize(expression: str) -> list[str]:
    """Split a selection expression while preserving glob and range syntax."""

    lexer = shlex.shlex(expression, posix=True, punctuation_chars="()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[str] = []
    for token in lexer:
        if token and all(char in "()" for char in token):
            tokens.extend(token)
        else:
            tokens.append(token)
    if not tokens:
        raise ValueError("Selection expression is empty.")
    return tokens


def _parse_integer_ranges(text: str, selector: str) -> set[int]:
    values: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"Empty value in {selector} selection: {text}")
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", item)
        if not match:
            raise ValueError(f"Invalid {selector} value or range: {item}")
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) is not None else start
        if stop < start:
            raise ValueError(f"Descending {selector} range is not allowed: {item}")
        values.update(range(start, stop + 1))
    return values


def _matches_patterns(value: str, raw_patterns: str) -> bool:
    patterns = [part.strip() for part in raw_patterns.split(",")]
    if any(not pattern for pattern in patterns):
        raise ValueError(f"Empty pattern in selection value: {raw_patterns}")
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


@dataclass
class _Parser:
    tokens: list[str]
    atoms: tuple[AtomRecord, ...]
    groups: dict[str, set[int]]
    position: int = 0

    @property
    def universe(self) -> set[int]:
        return set(range(len(self.atoms)))

    def current(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.current()
        if token is None:
            raise ValueError("Unexpected end of selection expression.")
        self.position += 1
        return token

    def parse(self) -> set[int]:
        result = self.parse_or()
        if self.current() is not None:
            raise ValueError(f"Unexpected token in selection: {self.current()}")
        return result

    def parse_or(self) -> set[int]:
        result = self.parse_and()
        while (token := self.current()) is not None and token.lower() == "or":
            self.take()
            result |= self.parse_and()
        return result

    def parse_and(self) -> set[int]:
        result = self.parse_not()
        while (token := self.current()) is not None and token.lower() == "and":
            self.take()
            result &= self.parse_not()
        return result

    def parse_not(self) -> set[int]:
        token = self.current()
        if token is not None and token.lower() == "not":
            self.take()
            return self.universe - self.parse_not()
        return self.parse_factor()

    def parse_factor(self) -> set[int]:
        token = self.take()
        lowered = token.lower()

        if token == "(":
            result = self.parse_or()
            if self.take() != ")":
                raise ValueError("Missing closing parenthesis in selection.")
            return result
        if token == ")":
            raise ValueError("Unexpected closing parenthesis in selection.")
        if lowered == "all":
            return self.universe
        if token.startswith("@"):
            return self._index_group(token[1:])
        if lowered in _SELECTOR_NAMES:
            value = self.take()
            if value.lower() in {"and", "or", "not"} or value in {"(", ")"}:
                raise ValueError(f"Selector '{token}' requires a value.")
            return self._selector(lowered, value)
        raise ValueError(
            f"Unknown selection term '{token}'. Supported selectors: "
            "all, name, resname, resid, chain, element, index, @Group."
        )

    def _index_group(self, name: str) -> set[int]:
        if not name:
            raise ValueError("An index group reference must have a name after '@'.")
        if name not in self.groups:
            available = ", ".join(sorted(self.groups)) or "none"
            raise ValueError(f"Unknown index group '@{name}'. Available groups: {available}")
        positions = self.groups[name]
        invalid = sorted(index for index in positions if index < 1 or index > len(self.atoms))
        if invalid:
            raise ValueError(
                f"Index group '@{name}' refers to atom {invalid[0]}, but the trajectory "
                f"contains {len(self.atoms)} atoms."
            )
        return {index - 1 for index in positions}

    def _selector(self, selector: str, value: str) -> set[int]:
        if selector == "resid":
            wanted = _parse_integer_ranges(value, selector)
            return {i for i, atom in enumerate(self.atoms) if atom.key.resid in wanted}
        if selector == "index":
            wanted = _parse_integer_ranges(value, selector)
            invalid = sorted(index for index in wanted if index < 1 or index > len(self.atoms))
            if invalid:
                raise ValueError(
                    f"Index selection refers to atom {invalid[0]}, but the trajectory "
                    f"contains {len(self.atoms)} atoms."
                )
            return {index - 1 for index in wanted}

        def field(atom: AtomRecord) -> str:
            if selector == "name":
                return atom.key.atomname
            if selector == "resname":
                return atom.resname
            if selector == "chain":
                return atom.key.chain
            if selector == "element":
                return atom.element
            raise AssertionError(selector)

        return {
            i for i, atom in enumerate(self.atoms)
            if _matches_patterns(field(atom), value)
        }


def resolve_selection(
    expression: str,
    atoms: tuple[AtomRecord, ...],
    groups: dict[str, set[int]] | None = None,
) -> np.ndarray:
    """Resolve an expression to sorted zero-based atom positions."""

    parser = _Parser(tokenize(expression), atoms, groups or {})
    selected = parser.parse()
    return np.asarray(sorted(selected), dtype=int)
