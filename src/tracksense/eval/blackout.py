"""
TrackSense GNSS Blackout Simulator (Person 4 - Step 12)

Reusable utility to simulate GNSS blackout intervals (signal loss) on timestamped
trajectory data (Pandas DataFrames and TrajectoryData objects).
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
from tracksense.eval.datatypes import TrajectoryData, BlackoutInterval


class GNSSBlackoutSimulator:
    """Simulator engine for injecting deterministic GNSS blackout intervals."""

    @staticmethod
    def simulate_blackout(
        data: Union[pd.DataFrame, TrajectoryData],
        blackout_start_time: float,
        blackout_duration: float,
        name: str = "",
        timestamp_col: str = "timestamp_s",
        gnss_available_col: str = "gnss_available",
        gnss_x_col: str = "gnss_x_m",
        gnss_y_col: str = "gnss_y_m",
    ) -> Union[pd.DataFrame, TrajectoryData]:
        """
        Simulates GNSS signal loss over [blackout_start_time, blackout_start_time + blackout_duration].
        
        Validation Rules:
        - blackout_duration must be > 0.
        - blackout_start_time must be within available data timestamp range.
        - blackout_start_time + blackout_duration must fit within available data timestamp range.
        - timestamps must be strictly ordered (monotonically increasing).
        - required timestamp column must exist.
        - input data object is NOT mutated (a copy is returned).
        
        Args:
            data: Input pd.DataFrame or TrajectoryData object
            blackout_start_time: Start timestamp in seconds
            blackout_duration: Duration in seconds (e.g., 30.0, 60.0, 120.0)
            name: Optional descriptive label for the blackout segment (e.g., "Tunnel 1")
            timestamp_col: Name of timestamp column in DataFrame
            gnss_available_col: Name of GNSS boolean availability column in DataFrame
            gnss_x_col: Optional GNSS X coordinate column in DataFrame
            gnss_y_col: Optional GNSS Y coordinate column in DataFrame
            
        Returns:
            A new instance of DataFrame or TrajectoryData with GNSS marked unavailable during blackout.
        """
        if blackout_duration <= 0:
            raise ValueError(
                f"Invalid blackout_duration ({blackout_duration}): duration must be > 0."
            )

        blackout_end_time = blackout_start_time + blackout_duration

        if isinstance(data, pd.DataFrame):
            return GNSSBlackoutSimulator._simulate_dataframe(
                df=data,
                start_time=blackout_start_time,
                end_time=blackout_end_time,
                timestamp_col=timestamp_col,
                gnss_available_col=gnss_available_col,
                gnss_x_col=gnss_x_col,
                gnss_y_col=gnss_y_col,
            )
        elif isinstance(data, TrajectoryData):
            return GNSSBlackoutSimulator._simulate_trajectory_data(
                trj=data,
                start_time=blackout_start_time,
                end_time=blackout_end_time,
                name=name,
            )
        else:
            raise TypeError(
                f"Unsupported data type {type(data)}: must be pd.DataFrame or TrajectoryData."
            )

    @staticmethod
    def _validate_timestamps(timestamps: np.ndarray, start_time: float, end_time: float):
        if len(timestamps) == 0:
            raise ValueError("Data contains no timestamps.")

        diffs = np.diff(timestamps)
        if np.any(diffs <= 0):
            raise ValueError("Timestamps are not strictly ordered (monotonically increasing).")

        min_t = float(timestamps[0])
        max_t = float(timestamps[-1])

        if start_time < min_t or start_time > max_t:
            raise ValueError(
                f"blackout_start_time ({start_time}s) is outside dataset timestamp range [{min_t:.2f}s, {max_t:.2f}s]."
            )

        if end_time > max_t + 1e-6:
            raise ValueError(
                f"Blackout end time ({end_time:.2f}s) exceeds maximum available timestamp ({max_t:.2f}s)."
            )

    @classmethod
    def _simulate_dataframe(
        cls,
        df: pd.DataFrame,
        start_time: float,
        end_time: float,
        timestamp_col: str,
        gnss_available_col: str,
        gnss_x_col: str,
        gnss_y_col: str,
    ) -> pd.DataFrame:
        if timestamp_col not in df.columns:
            raise KeyError(f"Required timestamp column '{timestamp_col}' not found in DataFrame.")

        timestamps = df[timestamp_col].to_numpy()
        cls._validate_timestamps(timestamps, start_time, end_time)

        # Work on a deep copy to guarantee input immutability
        out_df = df.copy(deep=True)

        blackout_mask = (timestamps >= start_time) & (timestamps <= end_time)

        # Mark GNSS availability false inside blackout, true outside
        out_df[gnss_available_col] = ~blackout_mask

        # If GNSS position columns exist, mask them with NaN inside blackout
        if gnss_x_col in out_df.columns:
            out_df.loc[blackout_mask, gnss_x_col] = np.nan
        if gnss_y_col in out_df.columns:
            out_df.loc[blackout_mask, gnss_y_col] = np.nan

        return out_df

    @classmethod
    def _simulate_trajectory_data(
        cls,
        trj: TrajectoryData,
        start_time: float,
        end_time: float,
        name: str,
    ) -> TrajectoryData:
        cls._validate_timestamps(trj.timestamps, start_time, end_time)

        new_interval = BlackoutInterval(
            start_time=start_time,
            end_time=end_time,
            name=name or f"Blackout_{start_time:.0f}s_{end_time:.0f}s",
        )

        blackout_mask = (trj.timestamps >= start_time) & (trj.timestamps <= end_time)

        new_gnss_pos = (
            np.copy(trj.gnss_position)
            if trj.gnss_position is not None
            else np.copy(trj.gt_position[:, :2])
        )
        new_gnss_pos[blackout_mask] = np.nan

        updated_intervals = list(trj.blackout_intervals) + [new_interval]

        return TrajectoryData(
            timestamps=np.copy(trj.timestamps),
            gt_position=np.copy(trj.gt_position),
            gt_velocity=np.copy(trj.gt_velocity) if trj.gt_velocity is not None else None,
            gnss_position=new_gnss_pos,
            imu_accel=np.copy(trj.imu_accel) if trj.imu_accel is not None else None,
            imu_gyro=np.copy(trj.imu_gyro) if trj.imu_gyro is not None else None,
            blackout_intervals=updated_intervals,
        )


def simulate_gnss_blackout(
    data: Union[pd.DataFrame, TrajectoryData],
    blackout_start_time: float,
    blackout_duration: float,
    name: str = "",
    timestamp_col: str = "timestamp_s",
    gnss_available_col: str = "gnss_available",
    gnss_x_col: str = "gnss_x_m",
    gnss_y_col: str = "gnss_y_m",
) -> Union[pd.DataFrame, TrajectoryData]:
    """Convenience wrapper for GNSSBlackoutSimulator.simulate_blackout."""
    return GNSSBlackoutSimulator.simulate_blackout(
        data=data,
        blackout_start_time=blackout_start_time,
        blackout_duration=blackout_duration,
        name=name,
        timestamp_col=timestamp_col,
        gnss_available_col=gnss_available_col,
        gnss_x_col=gnss_x_col,
        gnss_y_col=gnss_y_col,
    )
