from typing import Callable, Dict, Iterable
import jax
import jax.numpy as jnp
from jax import flatten_util
import jaxline
import haiku as hk
import matplotlib.pyplot as plt
import ml_collections
import optax
import pickle

import dataset
import experiment as exp
import models


# TODO(elpiloto): Fiddle with streamlit because it could be awesome and save
# your effin' life. Thanks, beyoundtwistedmeadows.


def load_checkpoint(fname: str) -> Dict:
  """Loads a checkpoint."""
  chk_pt = pickle.load(open(fname, 'rb'))
  if '_config' not in chk_pt:
    config = exp.get_config()
    chk_pt['_config'] = config
  return chk_pt


def restore_model(config: ml_collections.ConfigDict) -> Callable:
  model_cfg = config.experiment_kwargs.model
  model = models.get_model(model_cfg.name, model_cfg)
  model = hk.without_apply_rng(hk.transform(model))
  return model


def get_training_data(config: ml_collections.ConfigDict) -> Iterable:
  train_cfg = config.experiment_kwargs.data_config.train
  train_seed = config.experiment_kwargs.train_seed
  batch_size = config.experiment_kwargs.batch_size
  return dataset.build_train_data(train_cfg, train_seed, batch_size)


def get_grads(loss_fn, params, model, inputs, targets):
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(params, model, inputs, targets)
  flat_grads = {}
  for k, v in grads.items():
    name = k.replace("/", "")
    name = name.replace("~", "_")
    flat_grads[name], _ = flatten_util.ravel_pytree(v)
  return flat_grads


def main():
  chk_pt_file = './checkpoints/leafy-vortex-72_3000.pickle'
  chk_pt = load_checkpoint(chk_pt_file)
  config = chk_pt['_config']
  params = chk_pt['_params']
  model = restore_model(config)
  batch_iterator = get_training_data(config)
  inputs, targets = next(batch_iterator)
  loss = exp.mean_squared_error(params, model, inputs, targets)
  grad_fn = jax.grad(exp.mean_squared_error)
  grads = grad_fn(params, model, inputs, targets)
  flat_grads, _ = flatten_util.ravel_pytree(grads)


if __name__ == '__main__':
  main()


