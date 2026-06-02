"""Court geometry primitives used by tracking and analytics algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CourtGeometry:
    """Singles-court geometry in the project's V2 coordinate system."""

    singles_width: float = 8.23
    length: float = 23.77
    net_y: float = 0.0
    net_height: float = 0.914
    service_line_near: float = -6.40
    service_line_far: float = 6.40
    court_margin: float = 0.15

    @property
    def half_width(self) -> float:
        return self.singles_width / 2

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def x_min(self) -> float:
        return -self.half_width

    @property
    def x_max(self) -> float:
        return self.half_width

    @property
    def y_min(self) -> float:
        return -self.half_length

    @property
    def y_max(self) -> float:
        return self.half_length

    @property
    def baseline_near_max(self) -> float:
        return self.y_min + 5.0

    @property
    def baseline_far_min(self) -> float:
        return self.y_max - 5.0

    def side_for_y(self, y: float) -> str:
        return "near" if y < self.net_y else "far"

    def contains(self, x: float, y: float, *, margin: float | None = None) -> bool:
        if margin is None:
            margin = self.court_margin
        return (
            abs(x) <= self.half_width + margin
            and abs(y) <= self.half_length + margin
        )


DEFAULT_COURT = CourtGeometry()
