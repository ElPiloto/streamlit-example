"""Script to visually skim dataset.py."""
import numpy as np

from dataset import *


np.set_printoptions(suppress=True, precision=5)



# TODO(elpiloto): Parametrize this with flags.
if False:
  def integration():
    synthetic_generator = SyntheticGenerator()
    ds = BatchDataset(synthetic_generator.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=10).as_numpy_iterator()
  i = 0
  for i in range(1):
    inputs, targets = next(batch_iterator)
    print(inputs, targets)

if False:
  def integration():
    aoc_generator = AOCInputGenerator()
    ds = BatchDataset(aoc_generator.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=100).as_numpy_iterator()
  i = 0
  for i in range(10):
    inputs, targets = next(batch_iterator)
    print(inputs, targets)
