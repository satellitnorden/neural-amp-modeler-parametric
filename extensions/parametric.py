"""
Extension for training parametric models with NAM.
Registers a "parametric" dataset type that, instead of storing a single 'y_path', splits that into a 'data' list, with each entry having their own 'y_path' and parameters.
"""

try:
    from typing import Dict, Union, Tuple
    from copy import deepcopy
    from tqdm import tqdm
    import torch
    from nam.data import Dataset, ConcatDataset, register_dataset_initializer
    from nam.models.recurrent import LSTM, _LSTMHiddenCellType
    from nam.models.factory import register as RegisterModel
    from nam.train.lightning_module import LightningModule

    #The parametric dataset.
    class ParametricDataset(Dataset):
        def __init__(self, parameters: Dict[str, Union[bool, float, int]], *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._keys = tuple(k for k in parameters.keys())
            self._values = torch.Tensor([float(parameters[k]) for k in self._keys])

            #Fold the parameter values into 'x' directly here, so both training and plotting and such 'just works' later on. (:
            x = self._x

            if x.dim() == 1:
                x = x.unsqueeze(-1)

            parameter_values = self._values.clone()

            #TODO: Add some jitter. Try this someday if it makes a difference. (:
            """
            if torch.rand(()) < 0.3:
                sigma = 2e-5
                T = x.shape[0]

                noise = torch.randn(T, parameter_values.numel(), device=x.device) * sigma
                drift = noise.cumsum(dim=0)

                # weak attraction to original value (prevents runaway)
                alpha = 0.001
                param_traj = parameter_values + drift
                param_traj = (1 - alpha) * param_traj + alpha * parameter_values

                param_traj.clamp_(0.0, 1.0)
                parameter_tensor = param_traj
            else:
                parameter_tensor = parameter_values.expand(x.shape[0], -1)
            """

            parameter_tensor = parameter_values.expand(x.shape[0], -1)
            self._x = torch.cat([x, parameter_tensor], dim=-1)

        @classmethod
        def init_from_config(cls, config):
            config, data = cls.parse_config(config)
            datasets = []
            for _data in tqdm(data, desc="Parametric data gather..."):
                _config = deepcopy(config)
                y_path, parameters = [_data[k] for k in ("y_path", "parameters")]
                include = _data.get("include", True)
                _config.update(y_path=y_path, parameters=parameters)
                _config = super().parse_config(_config)

                if include:
                    datasets.append(ParametricDataset(**_config))

            return ConcatDataset(datasets)

        @classmethod
        def parse_config(cls, config):
            data = config.pop("data")
            return config, data

        @property
        def keys(self) -> Tuple[str]:
            return self._keys

        @property
        def values(self):
            return self._values
        	
    register_dataset_initializer("parametric", ParametricDataset.init_from_config)

    #Parametric LSTM class.
    class ParametricLSTM(LSTM):
        def __init(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
            #TODO: Figure out what to do here? We're getting a 1D tensor, we might want to pad it with default parameters, and run the model, but for now, output unity gain.
            return x

        def _get_initial_state(self, inputs=None) -> _LSTMHiddenCellType:
            """
            Convenience function to find a good hidden state to start the plugin at

            DX=input size
            L=num layers
            S=sequence length
            :param inputs: (1,S,DX)

            :return: (L,DH), (L,DH)
            """
            inputs = (
                torch.zeros((1, self._get_initial_state_burn_in, self._input_size))
                if inputs is None
                else inputs
            ).to(self.input_device)
            _, (h, c) = self._core(inputs)
            return h, c

    RegisterModel("ParametricLSTM", ParametricLSTM.init_from_config)

    print("'parametric' extension loaded!", flush=True)

except Exception as e:
    print("Failed to load 'parametric' extension!", flush=True)
    raise