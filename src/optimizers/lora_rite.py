"""LoRA-Rite optimizer adapted from https://arxiv.org/abs/2410.20625."""

import types

import torch
from torch.optim.optimizer import Optimizer


def create_preconditioner(x):
  # Only the rank-sized covariance needs double precision, not the LoRA factors.
  return torch.zeros((x.shape[-1], x.shape[-1]), dtype=torch.float64, device=x.device)


class _LoraRiteHelper:
  def __init__(self, maybe_inf_to_nan: bool = True):
    self._maybe_inf_to_nan = maybe_inf_to_nan
    self.reset_diagnostics()

  def reset_diagnostics(self):
    self.diagnostics = {
        "min_covariance_eigenvalue": 0.0,
        "roundoff_eigenvalues_clamped": 0,
        "skipped_factor_updates": 0,
        "zero_preconditioned_updates": 0,
    }

  def require_finite(self, value, name):
    if not torch.isfinite(value).all():
      raise FloatingPointError(f"LoRA-Rite nonfinite {name}")
    return value

  def inf_to_nan(self, array):
    """Do not silently turn an invalid optimizer calculation into a zero update."""
    return self.require_finite(array, "gradient or update")

  def bias_corrected_decay(self, step, decay: float):
    """Return the bias-corrected exponential decay."""
    t = step + 1.
    return decay * (1. - decay**(t - 1.)) / (1. - decay**t)

  def move_lora_dim_to_last(self, x, dim):
    x = torch.moveaxis(x, dim, -1)
    return x.reshape(-1, x.shape[-1]), x.shape

  def restore_original_shape_and_dim(self, x, dim, shape):
    x = x.reshape(shape)
    return torch.moveaxis(x, -1, dim)

  def restore_param_shape(self, x, p, dim):
    _, shape = self.move_lora_dim_to_last(p, dim)
    return self.restore_original_shape_and_dim(x, dim, shape)

  def spectral_preconditioner(self, x, esc, epsilon, relative_epsilon):
    """Evaluate the PSD spectral map without accepting materially negative modes.

    The active campaign rule is 1/(sqrt(lambda + escape) + epsilon).
    epsilon_root was unused in the original implementation; no additional ridge
    is introduced here. Relative mode scales epsilon by the covariance RMS scale.
    """
    self.require_finite(x, "covariance")
    source_epsilon = torch.finfo(x.dtype).eps
    covariance = self.make_symmetric(x.to(torch.float64))
    w, v = torch.linalg.eigh(covariance)
    self.require_finite(w, "covariance eigenvalues")
    scale = w.abs().max()
    tolerance = 8 * x.shape[-1] * source_epsilon * scale
    if (w < -tolerance).any():
      raise FloatingPointError("LoRA-Rite covariance is materially indefinite")
    minimum = float(w.min())
    self.diagnostics["min_covariance_eigenvalue"] = min(
        self.diagnostics["min_covariance_eigenvalue"], minimum)
    self.diagnostics["roundoff_eigenvalues_clamped"] += int((w < 0).sum())
    w = w.clamp_min(0)
    escape = torch.as_tensor(esc, dtype=w.dtype, device=w.device)
    self.require_finite(escape, "escape mass")
    if (escape < 0).any():
      raise FloatingPointError("LoRA-Rite escape mass must be nonnegative")
    regularizer = epsilon * torch.sqrt(scale) if relative_epsilon else epsilon
    denominator = torch.sqrt(w + escape) + regularizer
    # With epsilon=0, use the Moore-Penrose value on an exact zero mode.
    weights = torch.zeros_like(denominator)
    nonzero = denominator > 0
    weights[nonzero] = denominator[nonzero].reciprocal()
    self.require_finite(weights, "inverse-square-root weights")
    return v, weights

  def inverse_sqrt(self, x, esc, epsilon, epsilon_root, relative_epsilon=True, force_positive=False):
    # epsilon_root/force_positive remain accepted for source compatibility. PSD
    # roundoff is always handled, while genuine indefiniteness is always rejected.
    v, weights = self.spectral_preconditioner(x, esc, epsilon, relative_epsilon)
    return self.make_symmetric((v * weights) @ v.T).to(x.dtype)

  def make_symmetric(self, x):
    return (x + x.T) / 2

  def transform_first_moment_to_new_basis(
      self, m, p
  ):
    return m @ p.T

  def transform_second_moment_to_new_basis(
      self, v, p
  ):
    p = p.to(torch.float64)
    v_new = p @ v.to(torch.float64) @ p.T
    v_new = self.make_symmetric(v_new)
    return self.require_finite(v_new, "transported covariance")

  def get_unmagnified_rotate_second_escape(
      self, v_new, v_old
  ):
    eig_old = torch.linalg.eigvalsh(v_old)
    eig_new = torch.linalg.eigvalsh(v_new)
    eigen_diff = torch.maximum(torch.max(eig_old-eig_new), torch.tensor(0).to(eig_old))
    trace_diff = torch.maximum(torch.trace(v_old)-torch.trace(v_new), torch.tensor(0).to(eig_old))
    escape_mass = torch.minimum(eigen_diff, trace_diff)
    return escape_mass

  def get_unmagnified_grad(self, g, ri):
    return g@ri

  def rotate_update(self, g, ri):
    return g@ri.T

  def get_preconditioned_update(
      self,
      g,
      p,
      e,
      epsilon,
      epsilon_root,
      relative_epsilon,
      apply_escape,
      covariance_root=None,
  ):
    if not apply_escape:
      e = 0
    self.require_finite(g, "unmagnified gradient")
    v, weights = self.spectral_preconditioner(p, e, epsilon, relative_epsilon)
    if covariance_root is not None:
      self.require_finite(covariance_root, "covariance square root")
      _, singular, vh = torch.linalg.svd(covariance_root, full_matrices=False)
      v = vh.T
      scale = singular.max() if relative_epsilon else 1.0
      denominator = torch.sqrt(singular.square() + e) + epsilon * scale
      weights = torch.zeros_like(denominator)
      nonzero = denominator > 0
      weights[nonzero] = denominator[nonzero].reciprocal()
    # Applying the map in its eigenbasis avoids constructing an ill-conditioned
    # FP32 inverse matrix whose large null-space weights cancel in the product.
    u = ((g.to(torch.float64) @ v) * weights) @ v.T
    self.require_finite(u, "preconditioned update")
    self.diagnostics["zero_preconditioned_updates"] += int(torch.count_nonzero(u) == 0)
    return self.require_finite(u.to(g.dtype), "cast preconditioned update")

  def update_moment(self, step, update,
                     m, beta: float):
    beta_decay = self.bias_corrected_decay(step, beta)
    m = (1.0 - beta_decay) * update + beta_decay * m
    return m

  def update_moments(self, step, update,
                     m, v, beta1: float,
                     beta2: float):
    """Update the first and second moments."""
    beta1_decay = self.bias_corrected_decay(step, beta1)
    beta2_decay = self.bias_corrected_decay(step, beta2)
    m = (1.0 - beta1_decay) * update + beta1_decay * m
    v = (1.0 - beta2_decay) * (update**2) + beta2_decay * v
    return m, v

  def update_first_moments(
      self, step, update, moments, beta1: float
  ):
    """Update the first moment."""
    beta1_decay = self.bias_corrected_decay(step, beta1)
    s = (1.0 - beta1_decay) * update + beta1_decay * moments
    return s

  def compute_second_moments(
      self, update
  ):
    update = self.require_finite(update, "second-moment gradient").to(torch.float64)
    s = (update.T @ update) / update.shape[0]
    s = self.make_symmetric(s)
    return s

  def update_second_root(self, step, update, root, rotation, beta2):
    """QR square-root update, algebraically R.T R = beta*PVP.T + (1-beta)*G.T G/n.

    Keeping R avoids losing tiny singular directions by forming normal equations
    before applying the inverse square root. No ridge or rank cutoff is added.
    """
    decay = self.bias_corrected_decay(step, beta2)
    old = decay**0.5 * (root @ rotation.to(torch.float64).T)
    current = (1 - decay)**0.5 * update.to(torch.float64) / update.shape[0]**0.5
    return torch.linalg.qr(torch.cat((old, current)), mode="r").R

  def update_second_moments(
      self, step, update, moments, beta2: float
  ):
    """Update the second moment."""
    beta2_decay = self.bias_corrected_decay(step, beta2)
    s = (1.0 - beta2_decay) * update + beta2_decay * moments
    s = self.make_symmetric(s)
    return s

  def update_second_escape(
      self, step, update, moments, beta2: float,
  ):
    beta2_decay = self.bias_corrected_decay(step, beta2)
    return (1.0 - beta2_decay) * update + beta2_decay * moments

  def polar(self, m):
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    u = U @ Vh
    p = Vh.T @ S.diag() @ Vh
    p = self.make_symmetric(p)
    return  u.to(m.dtype), p.to(m.dtype)

  def get_rotation_and_basis(self, w):
    return torch.linalg.qr(w)

  def reduce_rms(self, x):
    return torch.sqrt(torch.mean(torch.pow(x, 2)))

  def clip_update(self, update, clip_threshold: float):
    mean_update = self.inf_to_nan(self.reduce_rms(update))
    denom = torch.maximum(torch.ones_like(mean_update), mean_update / clip_threshold)
    return update / denom

  def skip_update(self, update, skip_threshold: float):
    mean_update = self.inf_to_nan(self.reduce_rms(update))

    if mean_update > skip_threshold:
      update = torch.zeros_like(update)
      self.diagnostics["skipped_factor_updates"] += 1
    return update


class LoRARite(Optimizer):
  def __init__(
      self,
      params,
      betas,
      eps=1e-6,
      lr=1e-3,
      relative_epsilon: bool = False,
      clip_unmagnified_grad: float = 1.0,
      update_capping: float = 0.0,
      update_skipping: float = 1.0,
      weight_decay: float = 0.0,
      apply_escape: float = False,
      lora_l_dim: int = 0,
      lora_r_dim: int = -1,
      maybe_inf_to_nan: bool = True,
      balance_param: bool = False,
  ):
    beta1, beta2 = betas
    epsilon = eps
    epsilon_root = eps**2
    if lr < 0.0:
      raise ValueError(
          "Invalid learning rate: {} - should be >= 0.0".format(lr)
      )
    if not 0.0 <= beta1 < 1.0:
      raise ValueError(
          "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(
              beta1
          )
      )
    if not 0.0 <= beta2 < 1.0:
      raise ValueError(
          "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(
              beta2
          )
      )
    if not 0.0 <= epsilon:
      raise ValueError(
          "Invalid epsilon value: {} - should be >= 0.0".format(epsilon)
      )
    defaults = dict(
        lr=lr,
        weight_decay=weight_decay,
    )
    super().__init__(params, defaults)

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.epsilon_root = epsilon_root
    self.relative_epsilon = relative_epsilon
    self.apply_escape = apply_escape
    self.clip_unmagnified_grad = clip_unmagnified_grad
    self.update_capping = update_capping
    self.update_skipping = update_skipping
    self.weight_decay = weight_decay
    self.lora_l_dim = lora_l_dim
    self.lora_r_dim = lora_r_dim
    self.maybe_inf_to_nan = maybe_inf_to_nan
    self.balance_param = balance_param
    self.helper = _LoraRiteHelper()
    self.state_initialized = False
    self.count = 0
    self.last_diagnostics = {}

  def step(self, closure=None):
    helper = self.helper
    helper.reset_diagnostics()
    update_capping = self.update_capping
    update_skipping = self.update_skipping
    clip_unmagnified_grad = self.clip_unmagnified_grad
    weight_decay = self.weight_decay
    lora_l_dim = self.lora_l_dim
    lora_r_dim = self.lora_r_dim
    maybe_inf_to_nan = self.maybe_inf_to_nan
    balance_param = self.balance_param

    beta1 = self.beta1
    beta2 = self.beta2
    epsilon = self.epsilon
    epsilon_root = self.epsilon_root
    relative_epsilon = self.relative_epsilon
    apply_escape = self.apply_escape

    loss = None
    if closure is not None:
      loss = closure()

    self.count += 1

    if not self.state_initialized:
      self.state_initialized = True
      for group in self.param_groups:
        for p1, p2 in list(zip(group["params"], group["params"][1:]))[::2]:
          param_l, param_r = p1.data, p2.data

          self.state[p1]["attr"] = types.SimpleNamespace()
          state = self.state[p1]["attr"]
          state.step = 0

          param_l, _ = helper.move_lora_dim_to_last(param_l, lora_l_dim)
          param_r, _ = helper.move_lora_dim_to_last(param_r, lora_r_dim)

          state.v_l = create_preconditioner(param_l)
          state.v_r = create_preconditioner(param_r)
          state.root_v_l = torch.zeros_like(state.v_l)
          state.root_v_r = torch.zeros_like(state.v_r)
          state.m_l = torch.zeros_like(param_l)
          state.m_r = torch.zeros_like(param_r)
          state.basis_l = torch.zeros_like(param_l)
          state.basis_r = torch.zeros_like(param_r)
          state.escape_l = 0.0
          state.escape_r = 0.0

    g_norm = 0
    g_norm_sq = 0
    for group in self.param_groups:
      for p1, p2 in list(zip(group["params"], group["params"][1:]))[::2]:
        state = self.state[p1]["attr"]

        param_l, param_r = p1.data, p2.data
        param_l, _ = helper.move_lora_dim_to_last(param_l, lora_l_dim)
        param_r, _ = helper.move_lora_dim_to_last(param_r, lora_r_dim)

        decompose_l = helper.get_rotation_and_basis(param_l)
        decompose_r = helper.get_rotation_and_basis(param_r)
        basis_l = decompose_l[0]
        basis_r = decompose_r[0]
        rotate_l = decompose_l[1]
        rotate_r = decompose_r[1]
        rotate_inv_l = torch.linalg.pinv(rotate_l)
        rotate_inv_r = torch.linalg.pinv(rotate_r)
        p_l = basis_r.T @ state.basis_r
        p_r = basis_l.T @ state.basis_l

        update_l, update_r = helper.inf_to_nan(p1.grad.data), helper.inf_to_nan(p2.grad.data)
        update_l, _ = helper.move_lora_dim_to_last(update_l, lora_l_dim)
        update_r, _ = helper.move_lora_dim_to_last(update_r, lora_r_dim)
        update_l = helper.get_unmagnified_grad(update_l, rotate_inv_r)
        update_r = helper.get_unmagnified_grad(update_r, rotate_inv_l)

        if update_skipping > 0:
          update_l = helper.skip_update(update_l, update_skipping)
          update_r = helper.skip_update(update_r, update_skipping)

        state.basis_l = basis_l
        state.basis_r = basis_r

        state.rotate_inv_l = rotate_inv_l
        state.rotate_inv_r = rotate_inv_r
        state.update_l = update_l
        state.update_r = update_r
        state.p_l = p_l
        state.p_r = p_r

        g_norm_sq += torch.linalg.norm(update_l)**2
        g_norm_sq += torch.linalg.norm(update_r)**2
    g_norm = helper.require_finite(torch.sqrt(g_norm_sq), "unmagnified gradient norm")

    for group in self.param_groups:
      for p1, p2 in list(zip(group["params"], group["params"][1:]))[::2]:
        param_l, param_r = p1.data, p2.data
        param_l, _ = helper.move_lora_dim_to_last(param_l, lora_l_dim)
        param_r, _ = helper.move_lora_dim_to_last(param_r, lora_r_dim)

        state = self.state[p1]["attr"]
        count = state.step

        rotate_inv_l = state.rotate_inv_l
        rotate_inv_r = state.rotate_inv_r
        update_l = state.update_l
        update_r = state.update_r
        p_l = state.p_l
        p_r = state.p_r

        del state.rotate_inv_l
        del state.rotate_inv_r
        del state.update_l
        del state.update_r
        del state.p_l
        del state.p_r

        basis_l = state.basis_l
        basis_r = state.basis_r

        if clip_unmagnified_grad > 0:
          if g_norm > clip_unmagnified_grad:
            update_l = update_l/g_norm*clip_unmagnified_grad
            update_r = update_r/g_norm*clip_unmagnified_grad

        transformed_v_l = helper.transform_second_moment_to_new_basis(
            state.v_l,
            p_l
        )
        transformed_v_r = helper.transform_second_moment_to_new_basis(
            state.v_r,
            p_r
        )

        if apply_escape:
          escape_l = helper.get_unmagnified_rotate_second_escape(
              transformed_v_l,
              state.v_l,
          )

          escape_r = helper.get_unmagnified_rotate_second_escape(
              transformed_v_r,
              state.v_r,
          )

          escape_l = helper.update_second_escape(
              count, 0, escape_l + state.escape_l, beta2
          )
          escape_r = helper.update_second_escape(
              count, 0, escape_r + state.escape_r, beta2
          )
        else:
          escape_l = escape_r = 0

        root_v_l = helper.update_second_root(count, update_l, state.root_v_l, p_l, beta2)
        root_v_r = helper.update_second_root(count, update_r, state.root_v_r, p_r, beta2)
        v_l = helper.make_symmetric(root_v_l.T @ root_v_l)
        v_r = helper.make_symmetric(root_v_r.T @ root_v_r)

        update_l = helper.get_preconditioned_update(
            update_l,
            v_l,
            escape_l,
            epsilon,
            epsilon_root,
            relative_epsilon,
            apply_escape,
            covariance_root=root_v_l,
        )
        update_r = helper.get_preconditioned_update(
            update_r,
            v_r,
            escape_r,
            epsilon,
            epsilon_root,
            relative_epsilon,
            apply_escape,
            covariance_root=root_v_r,
        )

        m_l = helper.transform_first_moment_to_new_basis(
            state.m_l,
            p_l
        )
        m_r = helper.transform_first_moment_to_new_basis(
            state.m_r,
            p_r
        )

        m_l = helper.update_first_moments(count, update_l, m_l, beta1)
        m_r = helper.update_first_moments(count, update_r, m_r, beta1)

        update_l = m_l
        update_r = m_r

        if update_capping > 0:
          update_l = helper.clip_update(update_l, update_capping)
          update_r = helper.clip_update(update_r, update_capping)

        update_l = helper.rotate_update(update_l, rotate_inv_r)
        update_r = helper.rotate_update(update_r, rotate_inv_l)

        if weight_decay > 0:
          update_l = update_l + weight_decay * param_l
          update_r = update_r + weight_decay * param_r

        step_size = -1.0 * group["lr"]

        update_l = step_size * update_l
        update_r = step_size * update_r

        if balance_param:
          l_norm = torch.linalg.norm(param_l + update_l) + 1e-6
          r_norm = torch.linalg.norm(param_r + update_r) + 1e-6

          balanced_norm = torch.sqrt(l_norm * r_norm)

          update_l = update_l * (balanced_norm / l_norm) + param_l * (
              balanced_norm / l_norm - 1
          )
          update_r = update_r * (balanced_norm / r_norm) + param_r * (
              balanced_norm / r_norm - 1
          )

        update_l = helper.restore_param_shape(update_l, p1.data, lora_l_dim)
        update_r = helper.restore_param_shape(update_r, p2.data, lora_r_dim)
        helper.require_finite(update_l, "left parameter delta")
        helper.require_finite(update_r, "right parameter delta")
        p1.data.add_(update_l.to(p1.data.dtype))
        p2.data.add_(update_r.to(p2.data.dtype))
        helper.require_finite(p1.data, "left parameter")
        helper.require_finite(p2.data, "right parameter")

        state.step += 1
        state.v_l = v_l
        state.v_r = v_r
        state.root_v_l = root_v_l
        state.root_v_r = root_v_r
        state.m_l = m_l
        state.m_r = m_r
        state.escape_l = escape_l
        state.escape_r = escape_r

    self.last_diagnostics = dict(helper.diagnostics)
    self.last_diagnostics["unmagnified_gradient_norm"] = float(g_norm)
    return loss
