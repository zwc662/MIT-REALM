import matplotlib.pyplot as plt
import numpy as np
from attrs import define

from clfrl.dyn.dyn_types import State


class SegwayArtist(plt.Artist):
    @define
    class Params:
        pole_length: float
        wheel_radius: float

    def __init__(self, params: Params, pole_color: str = "C1", zorder: float = 8):
        super().__init__()
        self.params = params
        dummy_state = np.zeros(4)
        self.pole_color = pole_color
        self.wheel, self.pole = self._get_artists(dummy_state)
        self.set_zorder(zorder)

    @property
    def artists(self) -> list[plt.Artist]:
        return [self.wheel, self.pole]

    def set_figure(self, fig: plt.Figure):
        [artist.set_figure(fig) for artist in self.artists]

    def set_transform(self, t):
        [artist.set_transform(t) for artist in self.artists]

    def draw(self, renderer):
        [artist.draw(renderer) for artist in self.artists]

    def _get_pole_pts(self, state: State):
        p = self.params
        px, th, _, _ = state

        start_pt = np.array([px, p.wheel_radius])
        # theta=0 is up, positive is clockwise.
        dpos = np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)]) * p.pole_length
        end_pt = start_pt + dpos

        # (n_pts, 2)
        pole_pts = np.stack([start_pt, end_pt], axis=0)
        return pole_pts

    def _get_artists(self, state: State):
        p = self.params
        px, th, _, _ = state

        wheel_style = dict(color="0.1")
        pole_style = dict(color=self.pole_color)

        # Circle for the wheel.
        center = (px, p.wheel_radius)
        wheel = plt.Circle(center, radius=p.wheel_radius, **wheel_style)

        # Line for the pole.
        pole_pts = self._get_pole_pts(state)
        pole = plt.Line2D(pole_pts[:, 0], pole_pts[:, 1], **pole_style)

        return wheel, pole

    def update_state(self, state: np.ndarray):
        p = self.params
        px, th, _, _ = state
        center = (px, p.wheel_radius)
        self.wheel.set_center(center)

        pole_pts = self._get_pole_pts(state)
        self.pole.set_data(pole_pts[:, 0], pole_pts[:, 1])
        self.stale = True
