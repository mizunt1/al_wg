from dataclasses import dataclass, is_dataclass, fields, asdict, replace, field

@dataclass
class cmnist:
    data_mode: str = 'cmnist'
    model_name: str = 'BayesianNetRes50ULarger'
    lr: float = 1e-4
    batch_size: int = 130
    num_epochs: int = 50
    num_majority_points: int = 4000
    num_minority_points: int = 400 #400
    al_iters: int  = 100
    al_size: int = 10
    batch_size_test: int = 64
    n_maj_sources: int = 3
    num_classes: int = 2
    binary_classification: bool = True

@dataclass
class cmnist_ood:
    data_mode: str = 'cmnist_ood'
    model_name: str = 'BayesianNetRes50ULarger'
    lr: float = 1e-4
    batch_size: int = 130
    num_epochs: int = 50
    num_majority_points: int = 4000
    num_minority_points: int = 400
    al_iters: int  = 100
    al_size: int = 10
    batch_size_test: int = 64
    n_maj_sources: int = 3
    num_classes: int = 2
    binary_classification: bool = True

@dataclass
class cmnist_10:
    data_mode: str = 'cmnist'
    model_name: str = 'BayesianNetRes50ULarger'
    lr: float = 1e-4
    batch_size: int = 130
    num_epochs: int = 50
    num_majority_points: int = 20000
    num_minority_points: int = 2000
    al_iters: int  = 100
    al_size: int = 100
    batch_size_test: int = 64
    n_maj_sources: int = 3
    num_classes: int = 10
    binary_classification: bool = False

@dataclass
class wb:
    data_mode: str = 'wb'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-6
    batch_size: int = 2
    num_epochs: int = 30
    num_majority_points: int = 4000
    num_minority_points: int = 400
    al_iters: int  = 100
    al_size: int = 30
    batch_size_test: int = 2
    n_maj_sources: int = 3
    num_classes: int = 2

@dataclass
class celeba:
    data_mode: str = 'celeba'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-6
    batch_size: int = 2
    num_epochs: int = 60
    num_majority_points: int = 15000
    num_minority_points: int = 300
    al_iters: int  = 100
    al_size: int = 90
    batch_size_test: int = 2
    n_maj_sources: int = 7
    num_classes: int = 2

@dataclass
class fmow:
    data_mode: str = 'fmow'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-4
    batch_size: int = 20
    num_epochs: int = 30
    al_iters: int  = 100
    al_size: int = 6000
    batch_size_test: int = 20
    n_maj_sources: int = 4
    num_classes: int = 62
    group_proportions: list = field(default_factory=lambda: [0.2 for i in range(5)])
    max_training_data_size: int = None

@dataclass
class fmow_ood:
    data_mode: str = 'fmow_ood'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-4
    batch_size: int = 20
    num_epochs: int = 30
    al_iters: int  = 100
    al_size: int = 6000
    batch_size_test: int = 20
    n_maj_sources: int = 4
    num_classes: int = 62
    group_proportions: list = field(default_factory=lambda: [0.2 for i in range(5)])
    max_training_data_size: int = None

@dataclass
class camelyon:
    data_mode: str = 'camelyon'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-5
    batch_size: int = 10
    num_epochs: int = 50
    num_majority_points: int = 4000
    num_minority_points: int = 400
    al_iters: int  = 100
    al_size: int = 5
    batch_size_test: int = 10
    n_maj_sources: int = 4
    num_classes: int = 2
    group_proportions: list = field(default_factory=lambda: [0.2 for i in range(1)])
    max_training_data_size: int = None

@dataclass
class camelyon_ood:
    data_mode: str = 'camelyon_ood'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-5
    batch_size: int = 10
    num_epochs: int = 50
    num_majority_points: int = 4000
    num_minority_points: int = 400
    al_iters: int  = 100
    al_size: int = 5
    batch_size_test: int = 10
    n_maj_sources: int = 4
    num_classes: int = 2
    group_proportions: list = [1]
    group_proportions: list = field(default_factory=lambda: [0.2 for i in range(1)])
    max_training_data_size: int = None
    test_source: int = 0

def populate_args_from_dataclass(args, cfg):
    """
    For each field in the dataclass, if args.<field> is missing or None,
    populate it with cfg.<field>. Returns the modified args.
    """
    for f in fields(cfg):
        # field exists in argparse namespace?
        if hasattr(args, f.name):
            current = getattr(args, f.name)

            # Only fill in when argparse didn't specify a value
            if current is None:
                setattr(args, f.name, getattr(cfg, f.name))
        else:
            # argparse didn't define this attribute at all → create it
            setattr(args, f.name, getattr(cfg, f.name))

    return args

