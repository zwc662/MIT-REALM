import matplotlib.pyplot as plt
import numpy as np
from attrs import define


class QuadArtist(plt.Artist):
    @define
    class Params:
        # Half of wingspan.
        l: float
        # Height of rotor.
        rotor_height: float
        # Width of the prop.
        prop_width: float

    def __init__(self, state: np.ndarray, params: Params, zorder: float = 8):
        super().__init__()
        self.state = state
        self.params = params
        self.body, self.center = self._get_artists(state, params)

        self.set_zorder(zorder)

    @property
    def artists(self) -> list[plt.Artist]:
        return [self.body, self.center]

    def set_figure(self, fig: plt.Figure):
        [artist.set_figure(fig) for artist in self.artists]

    # def set_axes(self, axes: plt.Axes):
    #     [artist.set_axes(axes) for artist in self.artists]

    def set_transform(self, t):
        [artist.set_transform(t) for artist in self.artists]

    def draw(self, renderer):
        [artist.draw(renderer) for artist in self.artists]

    def _get_body_pts(self, state: np.ndarray, params: Params):
        p = params

        center = state[:2]
        theta = state[2]

        heading_vec = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
        tangent_vec = np.array([np.cos(theta), np.sin(theta)])
        left_pt = center - p.l * tangent_vec
        right_pt = center + p.l * tangent_vec

        left_pt2 = left_pt + p.prop_width * tangent_vec
        right_pt2 = right_pt - p.prop_width * tangent_vec

        left_pt_up = left_pt2 + p.rotor_height * heading_vec
        right_pt_up = right_pt2 + p.rotor_height * heading_vec

        left_pt_up_l = left_pt_up - p.prop_width * tangent_vec
        right_pt_up_l = right_pt_up - p.prop_width * tangent_vec

        right_pt_up_r = right_pt_up + p.prop_width * tangent_vec
        left_pt_up_r = left_pt_up + p.prop_width * tangent_vec

        body_pts = np.stack(
            [
                left_pt_up_l,
                left_pt_up_r,
                left_pt_up,
                left_pt2,
                left_pt,
                right_pt,
                right_pt2,
                right_pt_up,
                right_pt_up_l,
                right_pt_up_r,
            ],
            axis=0,
        )
        return body_pts

    def _get_artists(self, state: np.ndarray, params: Params):
        p = params

        bcenter_style = dict(color="black")
        body_style = dict(lw=1.5, color="blue")

        center = state[:2]
        body_center = plt.Circle(center, radius=p.l / 10, **bcenter_style)

        body_pts = self._get_body_pts(state, params)
        body = plt.Line2D(body_pts[:, 0], body_pts[:, 1], **body_style)

        # Draw body then body_center
        return body, body_center

    def update_state(self, state: np.ndarray):
        center = state[:2]
        self.center.set_center(center)

        body_pts = self._get_body_pts(state, self.params)
        self.body.set_data(body_pts[:, 0], body_pts[:, 1])
        self.stale = True
