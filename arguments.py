from dataclasses import dataclass, is_dataclass, fields, asdict, replace

@dataclass
class cmnist:
    data_mode: str = 'cmnist'
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

@dataclass
class celeba:
    data_mode: str = 'celeba'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-6
    batch_size: int = 2
    num_epochs: int = 60
    num_majority_points: int = 4000
    num_minority_points: int = 400
    al_iters: int  = 100
    al_size: int = 30
    batch_size_test: int = 2
    n_maj_sources: int = 3

@dataclass
class camelyon:
    data_mode: str = 'camelyon'
    model_name: str = 'BayesianNetDino'
    lr: float = 1e-6
    batch_size: int = 10
    num_epochs: int = 50
    num_majority_points: int = 4000
    num_minority_points: int = 400
    al_iters: int  = 100
    al_size: int = 10
    batch_size_test: int = 10
    n_maj_sources: int = 4

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

