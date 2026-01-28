"""
Extension for training parametric models with NAM.
Registers a "parametric" dataset type that, instead of storing a single 'y_path', splits that into a 'data' list, with each entry having their own 'y_path' and parameters.
"""

try:
    import numpy
    from typing import Dict, Union, Tuple, Optional
    from copy import deepcopy
    from tqdm import tqdm
    import torch
    from nam.data import Dataset, ConcatDataset, register_dataset_initializer
    from nam.models.recurrent import LSTM, _LSTMHiddenCellType
    from nam.models.wavenet import WaveNet, _WaveNet
    from nam.models.factory import register as RegisterModel
    from nam.train.lightning_module import LightningModule
    from matplotlib import pyplot as plt

    #The parametric dataset.
    class ParametricDataset(Dataset):
        def __init__(self, x, number_of_parameters, delay, start_seconds, stop_seconds, sample_rate, jitter, parameters: Dict[str, Union[bool, float, int]], *args, **kwargs):
            total_length = x.shape[0]

            super().__init__(x = x, delay = delay, start_seconds = start_seconds, stop_seconds = stop_seconds, sample_rate = sample_rate, *args, **kwargs)

            if (isinstance(parameters, str)):
                numpy_parameter_values = numpy.load(parameters);

                if False:
                    for _numpy_parameter_values in numpy_parameter_values:
                        plt.plot(_numpy_parameter_values[0:48000 * 4])
                        plt.show()

                params = torch.from_numpy(numpy_parameter_values).float()

                # Accept either (L, D) or (D, L)
                if params.shape[0] != number_of_parameters:
                    params = params.T

                self._parameters = params  # (D, L)
            elif isinstance(parameters, dict):
                keys = tuple(k for k in parameters.keys())
                self._parameters = torch.Tensor([float(parameters[k]) for k in keys])
                self._parameters = self._parameters[:, None].expand(-1, total_length)

            start = int(start_seconds * sample_rate) if start_seconds is not None else 0
            stop = int(stop_seconds * sample_rate) if stop_seconds is not None else None

            #print(f"Applying start: {start} and stop: {stop}")

            self._parameters = self._parameters[:, start:stop]

            if delay > 0:
                #print(f"Applying delay: {delay}")
                self._parameters = self._parameters[:, delay:]

            assert self._parameters.shape[1] == self._x.shape[0], "Mismatching shapes!"

            self._jitter = jitter

        @classmethod
        def init_from_config(cls, config):
            config, data, number_of_parameters = cls.parse_config(config)
            datasets = []
            for _data in tqdm(data, desc="Parametric data gather..."):
                _config = deepcopy(config)
                y_path, parameters = [_data[k] for k in ("y_path", "parameters")]
                jitter = _data.get("jitter", None)
                include = _data.get("include", True)
                _config.update(y_path=y_path, jitter=jitter, number_of_parameters=number_of_parameters, parameters=parameters)
                custom_x_path = _data.get("x_path", None)
                if custom_x_path is not None:
                    _config.update(x_path=custom_x_path)
                _config = super().parse_config(_config)

                if include:
                    datasets.append(ParametricDataset(**_config))

            return ConcatDataset(datasets)

        @classmethod
        def parse_config(cls, config):
            data = config.pop("data")
            number_of_parameters = len(config["parameters"])
            return config, data, number_of_parameters

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            :return:
                Parameter values (D,)
                Input (NX+NY-1,)
                Output (NY,)
            """
            x, y = super().__getitem__(idx)
            if idx >= len(self):
                raise IndexError(f"Attempted to access datum {idx}, but len is {len(self)}")
            i = idx * self._ny
            j = i + self.y_offset
            parameters = self._parameters[:, i : i + self._nx + self._ny - 1]

            if self._jitter is not None and self._jitter > 0.0:
                print(f"Applying jitter: {self._jitter}")
                jitter = ((torch.rand(parameters.shape[0], 1, device=parameters.device) - 0.5) * 2.0) * self._jitter
                jittered_parameters = torch.clamp(parameters + jitter, 0.0, 1.0)
                return jittered_parameters, x, y

            else:
                return parameters, x, y

        @property
        def parameters(self):
            return self._parameters
        	
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

    #_ParametricWaveNet class - The actual net class.
    class _ParametricWaveNet(_WaveNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x, c):
            y, head_input = x, None
            for layer in self._layers:
                head_input, y = layer(y, c, head_input=head_input)
            head_input = self._head_scale * head_input
            return head_input if self._head is None else self._head(head_input)

    #Parametric WaveNet class.
    class ParametricWaveNet(WaveNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._net = _ParametricWaveNet(*args, **kwargs)

        def forward(
            self,
            params: torch.Tensor,
            x: torch.Tensor,
            pad_start: Optional[bool] = None,
            **kwargs
        ):
            pad_start = self.pad_start_default if pad_start is None else pad_start
            if pad_start:
                x = torch.cat(
                    (torch.zeros((len(x), self.receptive_field - 1)).to(x.device), x), dim=1
                )

            return self._forward(params, x, **kwargs)

        def _forward(self, params, x):
            """
            params: (B, D) or (B, D, L)
            x: (B, L)
            """
            if x.ndim == 2:
                x = x[:, None, :]  # (B, 1, L)

            # Expand static params to time if needed
            if params.ndim == 2:
                params = params[..., None].expand(-1, -1, x.shape[2])

            y = self._net(x, params)
            return y[:, 0, :]

        def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
            #TODO: Figure out what to do here? We're getting a 1D tensor, we might want to pad it with default parameters, and run the model, but for now, output unity gain.
            return x

    RegisterModel("ParametricWaveNet", ParametricWaveNet.init_from_config)

    print("'parametric' extension loaded!", flush=True)

except Exception as e:
    print("Failed to load 'parametric' extension!", flush=True)
    raise