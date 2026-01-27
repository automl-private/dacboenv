from __future__ import annotations

from smac.runhistory.dataclasses import TrialInfo, TrialValue

from dacboenv.experiment.warmstart import WarmstartCallback

task_id = "dacbo_Cepisode_length_scaled_plus_logregret_AUCB-cont_Ssmart_Repisode_finished_scaled_Ibbob2d_3seeds"
trajectory_csv_fn = "runs/trajectory.csv"
callback = WarmstartCallback(task_id=task_id, trajectory_csv_fn=trajectory_csv_fn)


class FakeSMBO:
    def tell(self, info: TrialInfo, value: TrialValue, save: bool = False) -> None:
        pass


fake_smbo = FakeSMBO()
callback.on_start(smbo=fake_smbo)
