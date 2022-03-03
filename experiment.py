"""Jaxline experiment for solving Day 2, Part Advent of Code."""
import functools
import os
from typing import Dict, Optional

from absl import app
from absl import logging
from absl import flags
import pickle
import haiku as hk
from haiku import data_structures
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import wandb

import analysis
import dataset
import models
import utils

np.set_printoptions(suppress=True, precision=5)
tf.config.set_visible_devices([], 'GPU')
FLAGS = flags.FLAGS

# This may be unnecessary, try removing it when you have free time.
jax.config.update('jax_platform_name', 'gpu')
# Probably need it for precision.
# jax.config.update("jax_enable_x64", True)

run = wandb.init(project='aoc_2021_day2_part2', entity='elpiloto')



def get_config():
  # Common config to all jaxline experiments.
  config = base_config.get_base_config()
  config.training_steps = 100000
  config.checkpoint_dir = './checkpoints/'
  # Needed because jaxline version from pypi is broken and version from github
  # breaks everything else.
  config.train_checkpoint_all_hosts = False
  config.interval_type = 'steps'

  # Our config options
  exp = config.experiment_kwargs = ml_collections.ConfigDict()
  exp.train_seed = 107993
  exp.eval_seed = 8802
  exp.learning_rate = 1e-4
  exp.batch_size = 512

  optim = exp.optim = ml_collections.ConfigDict()
  optim.grad_clip_value = 10.

  exp.data_config = ml_collections.ConfigDict()
  train = exp.data_config.train = ml_collections.ConfigDict()
  train.min_pos = 0
  train.max_pos = 100
  train.min_magnitude = 0
  train.max_magnitude = 10
  train.min_aim = 0
  train.max_aim = 100

  eval = exp.data_config.eval = ml_collections.ConfigDict()
  eval.name = ["eval"]
  eval.min_pos = [0,]
  eval.max_pos = [2000,]
  eval.min_magnitude = [0,]
  eval.max_magnitude = [10,]
  eval.min_aim = [0,]
  eval.max_aim = [900,]

  model = exp.model = ml_collections.ConfigDict()
  model.name = 'nac_multi_nac'
  model.output_sizes = [16, 3]
  model.activation_fn = 'relu'
  model.magnitude_scale = 10.
  model.remove_pos = True
  wandb.config.update(exp.to_dict())
  return config


def mean_squared_error(params, model, inputs, targets):
  """Computes the mean squared error."""
  model_output = model.apply(params, inputs)
  # dimensions: [batch_size, 2]
  error = jnp.square(model_output - targets)
  summed = jnp.sum(error, axis=-1)
  # summed has shape: [batch_size]
  mse = jnp.mean(summed)
  return mse


def rounded_mean_squared_error(params, model, inputs, targets):
  """Computes the rounded mean squared error, not for training with!"""
  model_output = model.apply(params, inputs)
  model_output = jnp.round(model_output)
  # dimensions: [batch_size, 3]
  error = jnp.square(model_output - targets)
  pos_error = error[:, :2]
  aim_error = error[:, -1:]
  def sum_and_mse(x):
    summed = jnp.sum(x, axis=-1)
    mse = jnp.mean(summed)
    return mse

  return sum_and_mse(pos_error), sum_and_mse(aim_error), sum_and_mse(error), model_output


def show_model_predictions(inputs, targets, predictions, num_examples=1,
    from_end=False, silent=False):
  """Prints out model prediction versus ground truth."""
  this_example = {}
  for i in range(num_examples):
    if from_end:
      prediction = predictions[-(i+1)]
    else:
      prediction = predictions[i]
    target = targets[i]
    example = inputs[i]
    pos = example[:2]
    cmd_onehot = example[2:5]
    cmd_idx = np.argmax(cmd_onehot)
    cmd = dataset.Commands(cmd_idx).name
    magnitude = example[5]
    aim = example[6]
    if not silent:
      print(f'Input pos: {pos[0]}, {pos[1]}, {cmd} {magnitude}, {aim}\n-->  True: {target[0]}, {target[1]}, AIM: {target[2]},\n--> Model: {prediction[0]}, {prediction[1]}, AIM: {prediction[2]}')
    this_example = {
        'Input pos': f'{pos[0]}, {pos[1]}, {cmd} {magnitude}, {aim}',
        'True': f'{target[0]}, {target[1]}, AIM: {target[2]}',
        'Model': f'{prediction[0]}, {prediction[1]}, AIM: {prediction[2]}'
    }
  return this_example

class Experiment(experiment.AbstractExperiment):

  CHECKPOINT_ATTRS = {}
  NON_BROADCAST_CHECKPOINT_ATTRS = {
       '_params': '_params',
       '_opt_state': '_opt_state',
       '_config': '_config',
  }

  def __init__(self,
                mode: str,
                train_seed: int,
                eval_seed: int,
                learning_rate: float,
                batch_size: int,
                data_config: ml_collections.ConfigDict,
                model: ml_collections.ConfigDict,
                optim: ml_collections.ConfigDict,
                init_rng: Optional[jnp.DeviceArray] = None):
      super().__init__(mode, init_rng=init_rng)
      self._mode = mode
      self._train_seed = train_seed
      self._eval_seed = eval_seed
      self._learning_rate = learning_rate
      self._data_config = data_config
      self._batch_size = batch_size
      self._config = get_config()
      self._model_config = model
      self._optim_cfg = optim
      logging.log(logging.INFO, f'Launched experiment with mode = {mode}')
      run.tags += tuple(FLAGS.wandb_tags)
      self._counter = jnp.array([0.])

      # train and eval together
      if mode == 'train':
        # instantiate our training data
        self._train_data = self._build_train_data()
        # instantiate our evaluation data
        self._eval_datasets = self._build_eval_data()
        self._aoc_data = self._build_aoc_data()
        # instantiate our neural network
        model = self._initialize_model()
        train_inputs, _ = next(self._train_data)
        self._model = hk.without_apply_rng(hk.transform(model))
        self._params = self._model.init(
            init_rng,
            inputs=jnp.zeros_like(train_inputs)
        )
        # build our optimizer
        sched = optax.piecewise_constant_schedule(
            -self._learning_rate,
            {
              400: 0.1,
              800: 100.,
              1200: 0.01,
              1800: 100.,
              2400: 0.01,
              3200: 100.,
              3600: 0.01,
              4500: 100.,
              5200: 0.01,
              6000: 100.,
              7200: 0.01,
              8200: 100.,
              9000: 0.01,
              10200: 100.,
              11000: 0.01,
              12200: 100.,
              13000: 0.01,
              25000: 0.000001,
              35000: 0.0000001,
              45000: 0.00000001,
            }
        )
        # We put this in a optax schedule just for easy logging.
        self._sched = sched
        #opt = optax.adam(learning_rate=sched, b1=0.6, b2=0.6666)
        opt_chain = list()
        if self._optim_cfg.grad_clip_value > -1:
          opt_chain.append(optax.clip(self._optim_cfg.grad_clip_value))
        #opt_chain.append(optax.scale_by_adam())
        opt_chain.append(optax.scale_by_adam(b1=0.6, b2=0.6666))
        opt_chain.append(optax.scale_by_schedule(sched))

        opt = optax.chain(*opt_chain)
        self._opt_state = opt.init(self._params)
        # Example output, I just like to keep this.
        _ = self._model.apply(self._params, train_inputs)

        # build our update fn, which is called by our step function
        # Make update function.
        @jax.jit
        def update_fn(params, inputs, targets):
          loss, grads = jax.value_and_grad(mean_squared_error)(params, self._model, inputs,
              targets)
          updates, opt_state = opt.update(grads, self._opt_state, params)
          params = optax.apply_updates(params, updates)
          return params, opt_state, loss
        self._update_fn = update_fn

  def _initialize_model(self):
    return models.get_model(
        self._model_config.name,
        self._model_config
    )

  def _build_train_data(self):
    return dataset.build_train_data(
        self._data_config['train'],
        self._train_seed,
        self._batch_size)

  def _build_eval_data(self):
    """Builds eval data drawn from same distribution as training data."""
    ds_config = self._data_config['eval']
    datasets = {}
    eval_cfg_len = len(ds_config['name'])
    for i in range(eval_cfg_len):
      name = ds_config['name'][i]
      min_pos = ds_config['min_pos'][i]
      max_pos = ds_config['max_pos'][i]
      min_aim = ds_config['min_aim'][i]
      max_aim = ds_config['max_aim'][i]
      min_magnitude = ds_config['min_magnitude'][i]
      max_magnitude = ds_config['max_magnitude'][i]
      generator = dataset.SyntheticGenerator(
          min_pos=min_pos,
          max_pos=max_pos,
          min_magnitude=min_magnitude,
          max_magnitude=max_magnitude,
          min_aim=min_aim,
          max_aim=max_aim,
      )
      ds = dataset.BatchDataset(generator.generator())
      batch_iterator = ds(batch_size=self._batch_size).as_numpy_iterator()
      datasets[name] = batch_iterator
    return datasets

  def _build_aoc_data(self):
    # TODO(elpiloto): Figure out cleaner way of repeating dataset instead of
    # instantiating a new dataset each time
    generator = dataset.AOCInputGenerator()
    ds = dataset.BatchDataset(generator.generator())
    batch_iterator = ds(batch_size=100).as_numpy_iterator()
    return batch_iterator

  def step(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer:
      Optional[jl_utils.Writer]) -> Dict[str, np.ndarray]:

    is_logging_step = global_step % 300 == 0

    # Get next training example
    inputs, targets = next(self._train_data)

    params, opt_state, loss = self._update_fn(self._params, inputs, targets)

    learning_rate = self._sched(global_step)[0]
    scalars = {
        'loss': loss,
        'learning_rate': learning_rate,
    }
    if is_logging_step and global_step > 299:
      eval_scalars = self.evaluate(global_step=global_step, rng=rng, writer=writer)
      scalars.update(eval_scalars)

      grads = analysis.get_grads(
          mean_squared_error,
          self._params,
          self._model,
          inputs,
          targets,
      )
      utils.table_print(scalars, 'green', 'red')
      for k, v in grads.items():
        scalars[f'grads_{k}'] = wandb.Histogram(v)
      wandb.log(scalars, step=global_step)

    if global_step > 0 and global_step % 4000 == 0:
      self.save_state(global_step, chkpoint_name=run.name)

    self._params = params
    self._opt_state = opt_state

    return scalars

  def evaluate(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer:
      Optional[jl_utils.Writer]) -> Dict[str, np.ndarray]:

    errors = []
    pos_errors = []
    aim_errors = []
    for idx, (inputs, targets) in enumerate(self._aoc_data):
      pos_error, aim_error, error, predictions = rounded_mean_squared_error(self._params, self._model, inputs, targets)
      log_msg = {
          f'Error #{idx}': error,
          f'Aim Error #{idx}': aim_error,
          f'Pos Error #{idx}': pos_error,
      }
      from_end = idx == 9
      log_msg.update(
          show_model_predictions(inputs, targets, predictions,
            from_end=from_end, silent=True)
      )
      utils.table_print(log_msg, 'yellow', 'blue')
      errors.append(error)
      aim_errors.append(aim_error)
      pos_errors.append(pos_error)
    summed_error = np.sum(errors)
    pos_error = np.sum(pos_errors)
    aim_error = np.sum(aim_errors)
    self._aoc_data = self._build_aoc_data()
    
    return {
        'aoc_summed_error': summed_error,
        'aoc_pos_error': pos_error,
        'aoc_aim_error': aim_error,
        }


  def save_state(self, global_step, chkpoint_name='checkpoint'):
    snapshot_state = {}
    for attr_name, chk_name in self.NON_BROADCAST_CHECKPOINT_ATTRS.items():
      snapshot_state[chk_name] = getattr(self, attr_name)
    chk_file = f'{chkpoint_name}_{global_step[0]}.pickle'
    chk_path = os.path.join(self._config.checkpoint_dir, chk_file)
    with open(chk_path, mode='wb') as f:
      pickle.dump(snapshot_state, f)
    logging.log(
        logging.INFO,
        f'Saved checkpoint to: {chk_path} with keys: {snapshot_state.keys()}'
    )
  
  


if __name__ == '__main__':
  flags.DEFINE_list('wandb_tags', [], 'Tags to send to wandb.')
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
