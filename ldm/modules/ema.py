import copy
import torch

from torch import nn
from typing import Iterable


class LitEmaGnrl(nn.Module):

    def __init__(
        self,
        model:nn.Module,
        decay:float=0.9999,
        use_num_updates:bool=True,
    ):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0 if use_num_updates else -1

    def forward(
        self,
        model:nn.Module,
    ):
        decay = self.decay
        num_updates = self.num_updates
        if num_updates >= 0:
            num_updates += 1
            decay = min(self.decay, (1+num_updates)/(10+num_updates))
            self.num_updates = num_updates
        one_minus_decay = 1 - decay
        with torch.no_grad():
            m_new_ps = dict(model.named_parameters())
            m_old_ps = dict(self.model.named_parameters())
            for k in m_new_ps:
                m_old_ps[k].sub_(one_minus_decay*(m_old_ps[k]-m_new_ps[k]))

    def copy_to(
        self,
        model:nn.Module,
    ):
        m_new_ps = dict(model.named_parameters())
        m_old_ps = dict(self.model.named_parameters())
        for k in m_new_ps:
            m_new_ps[k].copy_(m_old_ps[k])

    def store(
        self,
        params:Iterable[nn.Parameter],
    ):
        self.cache_params = [param.clone().to('cpu') for param in params]

    def restore(
        self,
        params:Iterable[nn.Parameter],
    ):
        for c_p, n_p in zip(self.cache_params, params):
            n_p.copy_(c_p.to(n_p.device))


class LitEma(nn.Module):

    def __init__(
        self,
        model:nn.Module,
        decay:float=0.9999,
        use_num_upates:bool=True,
    ):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            'num_updates',
            torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({ name: s_name })
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model:nn.Module):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1+self.num_updates)/(10+self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay*(shadow_params[sname]-m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model:nn.Module):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters:Iterable[nn.Parameter]):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters:Iterable[nn.Parameter]):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
