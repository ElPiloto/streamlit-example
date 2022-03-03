from typing import Callable
import inspect
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections

import modules


def remove_pos(inputs: jnp.ndarray) -> jnp.ndarray:
  """Removes positions from an input."""
  return inputs[:, 2:]

def get_activation(name: str) -> jax.custom_jvp:
  """Gets jax.nn activation fn from string."""
  try:
    activation_fn = getattr(jax.nn, name)
  except:
    raise ValueError(f'Unknown activation function: {name}')
  return activation_fn


def mlp(config: ml_collections.ConfigDict) -> Callable:
  """Returns an MLP."""
  def forward(inputs):
    if config.remove_pos:
      inputs = remove_pos(inputs)
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(inputs)
    return outputs
  return forward


def skip_connection_mlp(config: ml_collections.ConfigDict) -> Callable:
  """Returns an MLP with a skip connection from input pos, aim to output."""
  def forward(inputs):
    orig_inputs = inputs
    if config.remove_pos:
      inputs = remove_pos(inputs)
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(inputs)
    pos = outputs[:, :2] + orig_inputs[:, :2]
    aim = outputs[:, -1:] + orig_inputs[:, -1:]
    outputs = jnp.concatenate([pos, aim], axis=-1)
    return outputs
  return forward


def skip_connection_mlp_scaled_magnitude(config: ml_collections.ConfigDict) -> Callable:
  """skip connection mlp that also scales command on input and output."""
  raise NotImplementedError('You need to fix the skip connection.')
  scale = config.magnitude_scale
  def forward(inputs):
    orig_inputs = inputs
    if config.remove_pos:
      inputs = remove_pos(inputs)
    # TODO(elpiloto): may be faster to just do a matmult with [0 0 0 0 0 1/scale.]
    new_inputs = jnp.concatenate(
        [inputs[:, :-1], inputs[:, -1:]/scale],
        axis=-1
    )
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(new_inputs)
    return (scale * outputs) + orig_inputs[:, :2]
  return forward


def nalu(config: ml_collections.ConfigDict) -> Callable:
  """Returns an NALU."""
  def forward(inputs):
    nalu = modules.NALU(output_size=config.output_sizes[-1], eps=1e-8)
    outputs = nalu(inputs)
    return outputs
  return forward


def nalu_mlp(config: ml_collections.ConfigDict) -> Callable:
  """Returns an NALU."""
  def forward(inputs):
    orig_inputs = inputs
    nalu = modules.NALU(output_size=config.output_sizes[-1]*3, eps=1e-8)
    nalu_out = nalu(inputs)

    if config.remove_pos:
      inputs = remove_pos(inputs)
    input_and_nalu_out = jnp.concatenate([nalu_out, inputs], axis=-1)
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(input_and_nalu_out)
    return outputs
  return forward


def nac_multi_nac(config: ml_collections.ConfigDict) -> Callable:
  """Returns an nac and multi_nac."""
  def forward(inputs):
    nac = modules.NAC(output_size=config.output_sizes[-1]*3)(inputs)
    multi_nac = modules.MultiNAC(output_size=config.output_sizes[-1]*3,
        eps=1e-8)(inputs)
    if config.remove_pos:
      inputs = remove_pos(inputs)
    combo_in = jnp.concatenate([nac, multi_nac, inputs], axis=-1)
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(combo_in)
    return outputs
  return forward


def get_model(model_name: str, config: ml_collections.ConfigDict) -> hk.Module:
  """Uses model_name to build a specific model from this module."""
  local_fns = {k: v for k, v in globals().items() if inspect.isfunction(v)}
  if model_name in local_fns:
    return local_fns[model_name](config)
  raise ValueError(f'Could not find `{model_name}` function in model.py.')

