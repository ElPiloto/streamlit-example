import jax
import jax.numpy as jnp
import haiku as hk
from haiku import data_structures as dstructs
import modules


def check_nac(output_size: int = 1, input_size: int = 2):
  """Checks modules.NAC"""

  def _forward(x):
    nac = modules.NAC(output_size=output_size)
    return nac(x)

  # this now has .init, .apply
  rng_seq = hk.PRNGSequence(111222)
  rng_key = rng_seq.next()

  model = hk.without_apply_rng(hk.transform(_forward))
  dummy_input = jnp.array([[4., -6.]])
  params = model.init(rng_key, x=dummy_input)

  mutable_params = dstructs.to_mutable_dict(params)
  def yolk_params(m_vals, w_vals):  # r/boneappletea
    mutable_params['nac']['m_hat'] = jnp.array(m_vals)
    mutable_params['nac']['w_hat'] = jnp.array(w_vals)
    return mutable_params

  # Expect W = [ 1, 1] = a + b
  print('Expecting 1*a + 1*b = 4 + -6 = -2')
  yolks = yolk_params([[100.], [100.]], [[100.], [100.]])
  model_output = model.apply(yolks, dummy_input)
  print(model_output)

  print('Expecting 1*a - 1*b = 4 - -6 = 10')
  yolks = yolk_params([[100.], [100.]], [[100.], [-100.]])
  model_output = model.apply(yolks, dummy_input)
  print(model_output)

  print('Expecting 0*a + 1*b = 0 + -6 = -6')
  yolks = yolk_params([[-100.], [100.]], [[100.], [100.]])
  model_output = model.apply(yolks, dummy_input)
  print(model_output)

  print('Expecting 0*a + 1*b = 0 + -6 = -6')
  yolks = yolk_params([[-100.], [100.]], [[-100.], [100.]])
  model_output = model.apply(yolks, dummy_input)
  print(model_output)


def check_nalu(output_size: int = 1, input_size: int = 2):
  """Checks modules.NALU"""

  def _forward(x):
    nac = modules.NALU(output_size=output_size)
    return nac(x)

  # this now has .init, .apply
  rng_seq = hk.PRNGSequence(111222)
  rng_key = rng_seq.next()

  model = hk.without_apply_rng(hk.transform(_forward))
  dummy_input = jnp.array([[4., 2]])
  params = model.init(rng_key, x=dummy_input)
  print(params)

  mutable_params = dstructs.to_mutable_dict(params)
  def yolk_params(m_vals, w_vals, g_vals):  # r/boneappletea
    mutable_params['nalu']['big_g'] = jnp.array(g_vals)
    mutable_params['nalu/~/nac']['m_hat'] = jnp.array(m_vals)
    mutable_params['nalu/~/nac']['w_hat'] = jnp.array(w_vals)
    return mutable_params

  print('Setting big g for addition.')
  # Expect W = [ 1, 1] = a + b
  print(f'Expecting 1*a + 1*b = {dummy_input[:, 0]} + {dummy_input[:, 1]} = {jnp.sum(dummy_input)}')
  yolks = yolk_params(
      m_vals=[[100.], [100.]],
      w_vals=[[100.], [100.]],
      g_vals=[[100.], [100.]]
  )
  model_output = model.apply(yolks, dummy_input)
  print(model_output)

  print('\n\nSetting big g for multiplication.')
  print(f'Expecting 1*a + 1*b = {dummy_input[:, 0]} + {dummy_input[:, 1]} = {jnp.sum(dummy_input)}')
  yolks = yolk_params(
      m_vals=[[100.], [100.]],
      w_vals=[[100.], [100.]],
      g_vals=[[-100.], [-100.]]
  )
  model_output = model.apply(yolks, dummy_input)
  print(model_output)


if __name__ == "__main__":
  check_nalu()


