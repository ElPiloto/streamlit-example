"""Dataset for Part 2, currently only suitable for MLPs, not RNNs."""
from enum import IntEnum
from typing import List, Iterable, Optional

import ml_collections
import numpy as np
import tensorflow as tf
from hard_coded import solution as hc_solution


AOC_INPUT_FILE = './aoc_input.txt'


class Commands(IntEnum):
  FORWARD = 0
  UP = 1
  DOWN = 2

  @classmethod
  def values(cls):
    return list(map(lambda c: c.value, cls))

CMD_TO_VECTOR = {
    Commands.FORWARD: np.array([1., 0., 0.]),
    Commands.UP: np.array([0., 0., -1.]),
    Commands.DOWN: np.array([0., 0., 1.]),
}


def get_new_state(pos: np.ndarray, cmd: Commands, magnitude: float, aim:
    float) -> np.ndarray:
  """Returns new position and aim after executing cmd."""
  delta = CMD_TO_VECTOR[cmd] * magnitude
  if cmd == Commands.FORWARD:
    delta[1] += aim * magnitude
  new_pos = pos + delta[:2]
  new_aim = aim + delta[2:]
  new_state = np.concatenate([new_pos, new_aim])
  return new_state


def command_idx_to_onehot(idx: int) -> np.ndarray:
  all_onehots = np.eye(3)
  command_onehot = np.squeeze(all_onehots[idx])
  return command_onehot


def solve_cumulative(lines: List[str]):
  inputs = []
  targets = []

  old_pos = np.zeros(shape=(2), dtype=np.float32)
  old_aim = np.zeros(shape=(1), dtype=np.float32)
  for l in lines:
    cmd, magnitude = l.split(' ')
    cmd = cmd.upper()
    cmd = Commands[cmd]
    cmd_idx = cmd.value
    cmd_onehot = command_idx_to_onehot(cmd_idx)
    magnitude = float(magnitude)
    this_input = np.concatenate([
      np.copy(old_pos),
      cmd_onehot,
      [magnitude, old_aim],
    ])
    inputs.append(this_input)
    new_state = get_new_state(old_pos, cmd, magnitude, old_aim)
    targets.append(new_state)
    new_pos = new_state[:2]
    new_aim = new_state[-1]
    old_pos = new_pos
    old_aim = new_aim
  return inputs, targets


class SyntheticGenerator():
  """Generates synthetic values."""

  def __init__(self,
      min_pos: int = 0,
      max_pos: int = 400,
      min_magnitude: int = 0,
      max_magnitude: int = 20,
      min_aim: int = 0,
      max_aim: int = 900,
      rng_seed: Optional[int] = 112233):
    self.rng_state = np.random.RandomState(rng_seed)
    self.min_pos = min_pos
    self.max_pos = max_pos
    self.min_magnitude = min_magnitude
    self.max_magnitude = max_magnitude
    self.min_aim = min_aim
    self.max_aim = max_aim

  def generator(self):
    def _generator():
      while True:
        # pick initial position
        pos = self.rng_state.randint(
            low = self.min_pos,
            high = self.max_pos,
            size=(2)
        )
        aim = self.rng_state.randint(
            low = self.min_aim,
            high = self.max_aim,
            size=(1)
        )

        # pick command
        unique_commands = len(Commands.values())
        command_idx = self.rng_state.randint(
            low = 0,
            high = unique_commands,
            size=(1)
        )
        command_onehot = command_idx_to_onehot(command_idx)

        # pick magnitude
        magnitude = self.rng_state.randint(
            low = self.min_magnitude,
            high = self.max_magnitude,
            size=(1)
        )
        # [x, y, FWD_binary, UP_binary, DOWN_binary, magnitude, aim]
        values = np.concatenate([pos, command_onehot, magnitude, aim])
        values = values.astype(np.float32)

        # Get new position
        new_state = get_new_state(
            values[:2],
            Commands(command_idx),
            values[-2],
            values[-1],
        )
        yield values, new_state
    return _generator


class AOCInputGenerator():

  def __init__(self, input_file: str = AOC_INPUT_FILE):
    lines = hc_solution.read_file(input_file)
    self._inputs, self._targets = solve_cumulative(lines)
    self._num_examples = len(self._inputs)

  def generator(self):
    def _generator():
      for i in range(self._num_examples):
        yield self._inputs[i], self._targets[i]
    return _generator


class BatchDataset:

  def __init__(self, generator):
    self._generator = generator

  def __call__(self, batch_size: int):
    ds = tf.data.Dataset.from_generator(
            self._generator,
            (tf.float32, tf.float32),
            output_shapes=((7,), (3,)),
    )
    ds = ds.batch(batch_size=batch_size)
    return ds

def build_train_data(train_config: ml_collections.ConfigDict, train_seed: int,
    batch_size: int) -> Iterable:
  min_pos = train_config['min_pos']
  max_pos = train_config['max_pos']
  min_magnitude = train_config['min_magnitude']
  max_magnitude = train_config['max_magnitude']
  min_aim = train_config['min_aim']
  max_aim = train_config['max_aim']
  generator = SyntheticGenerator(
      min_pos=min_pos,
      max_pos=max_pos,
      min_magnitude=min_magnitude,
      max_magnitude=max_magnitude,
      min_aim=min_aim,
      max_aim=max_aim,
      rng_seed=train_seed,
  )
  ds = BatchDataset(generator.generator())
  batch_iterator = ds(batch_size=batch_size).as_numpy_iterator()
  return batch_iterator
