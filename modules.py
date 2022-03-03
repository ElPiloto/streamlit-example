from typing import Optional
import jax
import jax.numpy as jnp
import haiku as hk


class NAC(hk.Module):
  """Implementation of Neural accumulator to add input values together."""

  def __init__(self, output_size: int, name: Optional[str] = None):
    super().__init__(name=name)
    self._output_size = output_size


  def __call__(self, x: jnp.ndarray):
    input_size = x.shape[-1]
    w_hat = hk.get_parameter("w_hat", shape=[input_size, self._output_size], dtype=x.dtype,
        init=hk.initializers.VarianceScaling())
    m_hat = hk.get_parameter("m_hat", shape=[input_size, self._output_size], dtype=x.dtype,
        init=hk.initializers.VarianceScaling())
    w = jnp.multiply(jax.nn.tanh(w_hat), jax.nn.sigmoid(m_hat))
    output = jnp.dot(x, w)
    return output


class MultiNAC(hk.Module):
  """Use sum of log in nac to do multiplication."""

  def __init__(self, output_size: int, eps: float = 1e-8, max_eps: bool = False, name: Optional[str] = None):
    super().__init__(name=name)
    self._output_size = output_size
    self._eps = eps
    self._max_eps = max_eps


  def __call__(self, x: jnp.ndarray):
    input_size = x.shape[-1]
    w_hat = hk.get_parameter("w_hat", shape=[input_size, self._output_size], dtype=x.dtype,
        init=hk.initializers.VarianceScaling())
    m_hat = hk.get_parameter("m_hat", shape=[input_size, self._output_size], dtype=x.dtype,
        init=hk.initializers.VarianceScaling())
    hk.initializers.RandomUniform(-0.1, 0.1)
    w = jnp.multiply(jax.nn.tanh(w_hat), jax.nn.sigmoid(m_hat))
    if self._max_eps:
      output = jnp.exp(jnp.dot(jnp.log(jnp.max(jnp.abs(x), self._eps), w)))
    else:
      output = jnp.exp(jnp.dot(jnp.log(jnp.abs(x) + self._eps), w))
    return output


class NALU(hk.Module):
  """Implementation of Neural Arithmetic Logic Unit to add or multiply input values together."""

  def __init__(self, output_size: int, eps: float = 1e-9, name: Optional[str] = None):
    super().__init__(name=name)
    self._output_size = output_size
    self._eps = eps
    self._nac = NAC(output_size, name='nac')

  def __call__(self, x: jnp.ndarray):
    input_size = x.shape[-1]
    big_g = hk.get_parameter("big_g", shape=[input_size, self._output_size], dtype=x.dtype,
        init=hk.initializers.VarianceScaling())
    g = jax.nn.sigmoid(jnp.dot(x, big_g))
    a = self._nac(x)
    m = jnp.exp(self._nac(jnp.log(jnp.abs(x) + self._eps)))
    output = jnp.multiply(g, a) + jnp.multiply(1. - g, m)
    return output

