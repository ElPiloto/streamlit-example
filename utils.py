"""Broad utility functions."""
from typing import Dict, Iterable
from termcolor import colored
from tabulate import tabulate


def table_print(d: Dict, col_color='green', table_color='red', silent: bool = False, floatfmt: str ='.3f',):
  """Prints a dict of keys as a table."""
  d2 = {}
  for k, v in d.items():
    k = colored(k, col_color)
    if not v is Iterable:
      d2[k] = [v]
    else:
      d2[k] = v
  table_str = colored(tabulate(
      d2,
      headers='keys',
      floatfmt=floatfmt,
      tablefmt='fancy_grid',
      stralign='center'
  ), table_color)
  if not silent:
    print(table_str)
  return table_str

