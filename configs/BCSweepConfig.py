import dataclasses

import yaml2pyclass


class BCSweepConfig(yaml2pyclass.CodeGenerator):
    @dataclasses.dataclass
    class LossParamsClass:
        pass

    @dataclasses.dataclass
    class BcBackboneParamsClass:
        out_channels: int

    @dataclasses.dataclass
    class SweepParamsClass:
        @dataclasses.dataclass
        class MetricClass:
            goal: str
            name: str

        @dataclasses.dataclass
        class ParametersClass:
            @dataclasses.dataclass
            class SeedClass:
                values: list

            @dataclasses.dataclass
            class DataSeedClass:
                values: list

            @dataclasses.dataclass
            class TrainTestValSplitClass:
                values: list

            @dataclasses.dataclass
            class LrClass:
                values: list

            @dataclasses.dataclass
            class BcBackboneClass:
                values: list

            @dataclasses.dataclass
            class EpochsClass:
                values: list

            @dataclasses.dataclass
            class LossClass:
                values: list

            @dataclasses.dataclass
            class BatchSizeClass:
                values: list

            @dataclasses.dataclass
            class SplinePathClass:
                values: list

            @dataclasses.dataclass
            class ObsClass:
                values: list

            seed: SeedClass
            data_seed: DataSeedClass
            train_test_val_split: TrainTestValSplitClass
            lr: LrClass
            bc_backbone: BcBackboneClass
            epochs: EpochsClass
            loss: LossClass
            batch_size: BatchSizeClass
            spline_path: SplinePathClass
            obs: ObsClass

        metric: MetricClass
        method: str
        parameters: ParametersClass

    name: str
    project: str
    entity: str
    device: str
    wandb_log_interval: int
    save_interval: int
    loss_params: LossParamsClass
    root_dir: str
    nb_datapoints: int
    resize: list
    spline_points: int
    shuffle: bool
    s: int
    bc_backbone_params: BcBackboneParamsClass
    sweep_params: SweepParamsClass
