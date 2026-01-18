"""SAWEI Policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import trim_mean

from dacboenv.policy.abstract_policy import AbstractPolicy
from dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


def sigmoid(x: float) -> float:
    """Sigmoid Function.

    Parameters
    ----------
    ----float
        Input.

    Returns
    -------
    float
        Sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def detect_adjust(
    UBR: list | np.ndarray,  # noqa: N803
    window_size: int = 10,
    atol_rel: float = 0.1,
    smooth: bool = True,  # noqa: FBT001, FBT002
    compute_gradient: bool = True,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """Signal the time to adjust the algorithm.

    First, smooth the UBR signal and then calculate the gradients.
    If the gradient is close to 0, signal time to adjust.

    Parameters
    ----------
    UBR : np.ndarray | list
        UBR history.
    window_size : int, optional
        Window size to smooth the UBR with, by default 10
    atol_rel : float, optional
        Relative absolute tolerance, by default 0.1.
        Is used to determine whether the smoothed UBR gradient is close to 0.
        Is the proportion of the current maximum of the gradient.
    smooth : bool, optional
        Whether to smooth the UBR signal, by default True.
    compute_gradient : bool, optional
        Whether to compute the gradient of the (smoothed) UBR signal, by default True

    Returns
    -------
    np.ndarray[bool]
        Adjust yes or no per UBR point.
    """
    UBR = np.array(UBR)[~np.isnan(np.array(UBR))]
    if len(UBR) == 1:
        return np.array([0])
    signal = apply_moving_iqm(U=UBR, window_size=window_size) if smooth else UBR

    if compute_gradient:
        signal = np.gradient(signal)

    # max_grad = np.maximum.accumulate(miqm_gradient)
    # adjust = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel*max_grad[i]) for i in range(len(miqm_gradient))])
    # adjust[0] = 0  # misleading signal bc of iqm

    G_abs = np.abs(signal)
    max_grad = [np.nanmax(G_abs[: i + 1]) for i in range(len(G_abs))]
    adjust = np.array([np.isclose(signal[i], 0, atol=atol_rel * max_grad[i]) for i in range(len(signal))])
    # adjust = np.isclose(miqm_gradient, 0, atol=1e-5)
    adjust[:window_size] = 0  # misleading signal bc of iqm

    return adjust


# Moving IQM
def apply_moving_iqm(U: np.ndarray | list, window_size: int = 5) -> np.ndarray:  # noqa: N803
    """Moving IQM for UBR.

    Smoothes the noisy UBR signal.

    Parameters
    ----------
    U : np.ndarray | list
        UBR history.
    window_size : int, optional
        The window size for smoothing, by default 5.

    Returns
    -------
    np.ndarray
        Smoothed UBR.
    """

    def moving_iqm(X: np.ndarray) -> float:  # noqa: N803
        """Apply the IQM to one slice (X) of the UBR.

        Parameters
        ----------
        X : np.ndarray
            One slice of the UBR.

        Returns
        -------
        float
            IQM of this slice.
        """
        return trim_mean(X, 0.25)

    # Pad UBR so we can apply the sliding window
    U_padded = np.concatenate((np.array([U[0]] * (window_size - 1)), U))
    # Create slices to apply our smoothing method
    slices = sliding_window_view(U_padded, window_size, axis=0)
    # Apply smoothing
    return np.array([moving_iqm(s) for s in slices])


class SAWEIPolicy(AbstractPolicy):
    """Policy that implements SAWEI."""

    def __init__(
        self,
        env: DACBOEnv,
        alpha: float = 0.5,
        delta: float = 0.1,
        window_size: int = 7,
        atol_rel: float = 0.1,
        track_attitude: str = "last",
        bounds: tuple[float, float] | None = None,
        auto_alpha: bool = False,  # noqa: FBT001, FBT002
    ):
        """Initialize the SAWEI policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        alpha : float, optional
            The initial weight of weighted expected improvement, by default 0.5.
            This equals EI.
        delta : float | str, optional
            The additive magnitude of change, by default 0.1.
            This is added or subtracted to the curent alpha.
            The sign will be determined by the algorithm and is opposite to the
            current search attitude.
            Delta can also be "auto" which equals to auto_alpha=True. Experimental.
        window_size : int, optional
            Window size to smooth the UBR signal, by default 7.
            We smooth the UBR because we observed it to be very noisy from step to step.
        atol_rel : float, optional
            The relative absolute tolerance, by default 0.1.
            atol_rel is used to check whether the gradient of the smoothed UBR is
            approximately zero. The bigger atol_rel, the more often we should
            adjust. The absolute tolerance is determined by the current maximum
            gradient times this parameter.
        track_attitude : str, optional
            How far the search attitude is tracked, by default "last".
            Following options are available:
            - last: Only compare the WEI terms from the last optimization step. This
                worked best in the experiments.
            - until_inc_change: The WEI terms are tracked from the last time the incumbent
                changed.
            - until_last_adjust: The WEI terms are tracked from the last time SAWEI
                self-adjusted alpha, the exploration-exploitation trade-off.
        bounds : bool, optional
            The lower and upper bound for alpha. Defaults to (0, 1).
        auto_alpha : bool, optional
            By default False. Experimental feature. If set to true, directly determine
            alpha based on the distance between the exploration and exploitation summands.
            Empirically did not work that well.
        """
        super().__init__(env)
        self._alpha = alpha
        self._delta = delta
        self._window_size = window_size
        self._atol_rel = atol_rel
        self._track_attitude = track_attitude
        self._auto_alpha = auto_alpha

        self._last_inc_count: int = 0
        self._pi_term_sum: float = 0.0
        self._ei_term_sum: float = 0.0
        if bounds is None:
            self._bounds = (0.0, 1.0)
        else:
            self._bounds = bounds
        self._history: list[dict[str, Any]] = []

        assert isinstance(self._env._smac_instance._intensifier._config_selector._acquisition_function, WEI)

    def __call__(self, obs: ObsType) -> ActType:  # noqa: C901, PLR0912, PLR0915
        """Return an action based on SAWEI.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation.

        Returns
        -------
        ActType
            The action determined by SAWEI.
        """
        solver = self._env._smac_instance

        smooth = True
        compute_gradient = True

        if "ubr" in obs:
            signal = obs["ubr"]
            compute_gradient = True
            smooth = True
        elif "ubr_smoothed_gradient" in obs:
            signal = obs["ubr_smoothed_gradient"]
            compute_gradient = False
            smooth = False
        elif "ubr_gradient" in obs:
            signal = obs["ubr_gradient"]
            compute_gradient = False
            smooth = False
        else:
            raise ValueError("No UBR information in observation.")

        signal = signal.item()
        state = {
            "n_evaluated": solver.runhistory.finished,
            "alpha": self._alpha,
            "n_incumbent_changes": int(solver._intensifier._incumbents_changed),
            "wei_ei_term": obs["acq_value_WEI_explore"].item(),
            "wei_pi_term": obs["acq_value_PI"].item(),  # pure PI according to SAWEI paper
            "ubr": signal,
        }

        adjust = False
        UBR = [s["ubr"] for s in self._history]

        # We need at least 2 UBRs to compute the gradient
        if len(UBR) >= 2:  # noqa: PLR2004
            adjust = detect_adjust(
                UBR=UBR,
                window_size=self._window_size,
                atol_rel=self._atol_rel,
                smooth=smooth,
                compute_gradient=compute_gradient,
            )[-1]

        self._pi_term_sum += state["wei_pi_term"]
        self._ei_term_sum += state["wei_ei_term"]

        if adjust:
            if self._track_attitude == "last":
                # Calculate attitude: Exploring or exploiting?
                # Exploring = when ei term is bigger
                # Exploiting = when pi term is bigger
                exploring = state["wei_pi_term"] <= state["wei_ei_term"]
                distance = state["wei_ei_term"] - state["wei_pi_term"]
            elif self._track_attitude in ["until_inc_change", "until_last_adjust"]:
                exploring = self._pi_term_sum <= self._ei_term_sum
                distance = self._ei_term_sum - self._pi_term_sum
            else:
                raise ValueError(f"Unknown track_attitude {self._track_attitude}.")
            if self._auto_alpha:
                alpha = sigmoid(distance)
            else:
                # If attitude is
                # - exploring (exploring==True): increase alpha, change to exploiting
                # - exploiting (exploring==False): decrease alpha, change to exploring
                sign = 1 if exploring else -1
                alpha = self._alpha + sign * self._delta  # type: ignore[operator]

            # Bound alpha
            lb, ub = self._bounds
            self._alpha = max(lb, min(ub, alpha))

        if self._track_attitude == "until_inc_change":
            if state["n_incumbent_changes"] > self._last_inc_count:
                self._last_inc_count = state["n_incumbent_changes"]  # type: ignore[assignment]
                self._pi_term_sum = 0.0
                self._ei_term_sum = 0.0
        elif self._track_attitude == "until_last_adjust" and adjust:
            self._pi_term_sum = 0.0
            self._ei_term_sum = 0.0

        self._history.append(state)

        return self._alpha
