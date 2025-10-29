"""TVAE module."""

import numpy as np
import torch
import time
import os
import copy
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


def delete_first_n(ordered_dict, n):
    """
    Delete the first n key-value pairs from an ordered dictionary.
    
    Parameters:
        ordered_dict (OrderedDict): The ordered dictionary from which to delete key-value pairs.
        n (int): The number of key-value pairs to delete.
        
    Returns:
        OrderedDict: The updated ordered dictionary.
    """
    keys_list = list(ordered_dict.keys())
    for i in range(min(n, len(keys_list))):
        del ordered_dict[keys_list[i]]
    return ordered_dict

def delete_last_n(ordered_dict, n):
    """
    Delete the last n key-value pairs from an ordered dictionary.
    
    Parameters:
        ordered_dict (OrderedDict): The ordered dictionary from which to delete key-value pairs.
        n (int): The number of key-value pairs to delete.
        
    Returns:
        OrderedDict: The updated ordered dictionary.
    """
    keys_list = list(ordered_dict.keys())
    for i in range(-1, -n-1, -1):
        del ordered_dict[keys_list[i]]
    return ordered_dict


class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        max_train_time=None,
        pretrain_model_path=None
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        
        self.max_train_time = max_train_time
        self.pretrain_model_path = pretrain_model_path

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        print("Start data transformation...")
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        print("Data transformation finished!")

        data_dim = self.transformer.output_dimensions
        print("Data dimension: ", data_dim)
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)

        # Load pretrained model if path is provided
        if self.pretrain_model_path is not None:
            if not os.path.exists(self.pretrain_model_path):
                raise ValueError(f"Pretrained model path {self.pretrain_model_path} does not exist!")
        
            pretrain_state_dict = torch.load(self.pretrain_model_path)
            print(dir(pretrain_state_dict))
            '''
            Encoder:
                seq.0.weight torch.Size([128, 227])
                seq.0.bias torch.Size([128])
                seq.2.weight torch.Size([128, 128])
                seq.2.bias torch.Size([128])
                fc1.weight torch.Size([128, 128])
                fc1.bias torch.Size([128])
                fc2.weight torch.Size([128, 128])
                fc2.bias torch.Size([128])
            '''
            pretrain_encoder_state_dict = delete_first_n(copy.deepcopy(pretrain_state_dict.encoder.state_dict()), 2)
            print("Encoder pretrained state dict:")
            for k, v in pretrain_encoder_state_dict.items():
                print(k, v.shape)
            self.encoder.load_state_dict(pretrain_encoder_state_dict, strict=False)
            
            '''
            Decoder:
                sigma torch.Size([227])
                seq.0.weight torch.Size([128, 128])
                seq.0.bias torch.Size([128])
                seq.2.weight torch.Size([128, 128])
                seq.2.bias torch.Size([128])
                seq.4.weight torch.Size([227, 128])
                seq.4.bias torch.Size([227])
            '''
            pretrain_decoder_state_dict = delete_last_n(copy.deepcopy(pretrain_state_dict.decoder.state_dict()), 2)
            pretrain_decoder_state_dict = delete_first_n(pretrain_decoder_state_dict, 1)
            print("Decoder pretrained state dict:")
            for k, v in pretrain_decoder_state_dict.items():
                print(k, v.shape)
            self.decoder.load_state_dict(pretrain_decoder_state_dict, strict=False)

            print(f"Pretrained model loaded from {self.pretrain_model_path}")

        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        start_train_time = time.time()
        stop_training_flag = False
        for i in range(self.epochs):
            if stop_training_flag:
                break
            for id_, data in enumerate(loader):
                time_elapsed = time.time() - start_train_time
                if self.max_train_time is not None and time_elapsed > self.max_train_time:
                    print(f"{time_elapsed} seconds elapsed... Max training time {self.max_train_time} reached. Stopping training.")
                    stop_training_flag = True
                    break
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
