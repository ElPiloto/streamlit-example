from typing import List, Optional, Tuple
import numpy as np
from tabulate import tabulate


INPUT_FILE = '../aoc_input.txt'
cmd_to_vector = {
    'forward': np.array([1., 0., 0.], dtype=np.float64),
    'up': np.array([0., 0., -1.], dtype=np.float64),
    'down': np.array([0., 0., 1.], dtype=np.float64),
}

def parse_line(x: str) -> Tuple[str, int]:
  """Returns command and magnitude from line."""
  cmd, magnitude = x.split(' ')
  return cmd, int(magnitude)


def read_file(fname: Optional[str] = INPUT_FILE) -> List[str]:
  with open(fname) as file:
    lines = [line.rstrip() for line in file]
  return lines


def solve(lines: List[str]):
  # pos.x, pos.y, aim
  state = np.zeros(shape=(3), dtype=np.float64)
  max_aim = -1
  for idx, l in enumerate(lines):
    cmd, magnitude = parse_line(l)
    delta = cmd_to_vector[cmd] * magnitude
    if cmd == 'forward':
      aim = state[-1]
      delta += np.array([0., aim * magnitude, 0.], np.float64)
    state += delta
    max_aim = max(max_aim, state[-1])
    print(tabulate({'#': [idx], 'pos': [state[:2]], 'aim': [state[-1]]},
      headers='keys'))
    print('\n')
  # TODO(elpiloto): This will mess you up if you don't fix it before printing
  # final answer.
  product = np.prod(state[0:2])
  # 1954293920 was the correct answer, required more precision than np.float32
  print(f'Final position: {state}, product: {product}, max aim: {max_aim}')


if __name__ == '__main__':
  lines = read_file()
  solve(lines)
