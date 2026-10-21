"""Choosing a model, or a chain of models, for a requested upscale factor.

Models have a fixed native integer factor, so selection is a matching problem
rather than a configuration one. The rules, in order of preference:

1. An entry whose factor equals the request wins outright; no resample is
   needed and quality is at its best.
2. Otherwise the smallest entry that overshoots the request is used, and the
   result is resampled down to the exact target afterwards.
3. If nothing reaches the request, entries are chained whose factors multiply up
   to cover it, for example x4 then x4 to reach x16.
4. If even a chain cannot reach it, the largest entry is used and the caller is
   warned that the remainder will be a plain resample.

By default only general purpose models are considered. A model trained for a
narrower domain carries a tag other than the default and is therefore never
selected unless the caller explicitly asks for that tag. This keeps a routine
photo request from silently being handed to a model tuned for something else.
"""

from srcli.models import registry

# A model is eligible for automatic selection only if it carries this tag.
# Narrower models deliberately omit it so they can never be picked by accident,
# no matter how a request is phrased.
DEFAULT_TAG = "general"


def _candidates(tags):
  """Return the entries eligible for this request.

  With no tags the general purpose models are returned. With tags, only entries
  carrying at least one of them are returned, which is how a caller opts into a
  narrower model on purpose.
  """
  entries = [registry.REGISTRY[name] for name in registry.names()]
  wanted = set(tags) if tags else {DEFAULT_TAG}
  return [entry for entry in entries if wanted & set(entry.tags)]


def _chain_for(target, scales):
  """Find the shortest chain of factors whose product covers `target`.

  This is a small search over the available factors. Reusing the largest factor
  first keeps the number of inference passes, and therefore the runtime, low.
  The search is bounded because a handful of distinct factors is all any
  registry ever holds.
  """
  # Sort descending so the greedy choice starts from the biggest step.
  ordered = sorted(scales, reverse=True)
  best = None
  # Depth first search over the distinct factors, pruning chains that are no
  # better than one already found.
  stack = [(1, [])]
  while stack:
    product, chain = stack.pop()
    if best is not None and len(chain) >= len(best):
      continue
    if product >= target:
      if best is None or len(chain) < len(best):
        best = chain
      continue
    for factor in ordered:
      # A chain is only useful while it stays at or below the target; going over
      # is allowed once, and handled as the final overshoot.
      if product * factor <= target:
        stack.append((product * factor, chain + [factor]))
      else:
        candidate = chain + [factor]
        if best is None or len(candidate) < len(best):
          best = candidate
  return best


def select(target, tags=None):
  """Choose how to reach `target` using the registered models.

  Returns a dict describing the plan: the ordered list of model names to run,
  the combined factor they provide, and whether a final resample is needed to
  land exactly on the requested size. `tags` opts into specialised models; when
  it is omitted only the general purpose models are considered.
  """
  entries = _candidates(tags)
  if not entries:
    raise ValueError("no models available to satisfy the request")

  scales = sorted({entry.scale for entry in entries})

  # Exact match first: a single model at the requested factor.
  exact = [entry for entry in entries if entry.scale == target]
  if exact:
    return _plan([_pick(exact).name], target, target)

  # Single model that overshoots: least overshoot keeps quality loss smallest.
  larger = sorted([entry for entry in entries if entry.scale > target],
                  key=lambda e: e.scale)
  if larger:
    return _plan([larger[0].name], larger[0].scale, target)

  # No single model reaches the target, so try to chain factors.
  chain = _chain_for(target, scales)
  if chain:
    names = []
    product = 1
    for factor in chain:
      # Pick one model per factor, deterministically. The chain deliberately
      # reuses the same model when a factor appears more than once; swapping in
      # a different model of the same factor would change the look halfway
      # through the image for no benefit.
      names.append(_pick([entry for entry in entries if entry.scale == factor]).name)
      product *= factor
    return _plan(names, product, target)

  # Nothing can reach the target: clamp to the largest model and let the final
  # resample make up the difference.
  largest = max(entries, key=lambda entry: entry.scale)
  plan = _plan([largest.name], largest.scale, target)
  plan["warn"] = (
      f"largest available model is x{largest.scale}; reaching x{target} needs a plain "
      "resample for the remainder")
  return plan


def _pick(entries):
  """Choose one entry from a group, ordered by name so the result is stable."""
  return min(entries, key=lambda entry: entry.name)


def _plan(names, factor, target):
  """Assemble the plan dict, recording whether a final resample is required."""
  return {
      "models": names,
      "factor": factor,
      "target": target,
      "needs_resize": factor != target,
  }
