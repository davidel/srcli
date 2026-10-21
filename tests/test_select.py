from srcli.models import select


def test_select_exact_factor_uses_a_single_model_without_resize():
  plan = select.select(4)
  assert plan["factor"] == 4
  assert len(plan["models"]) == 1
  assert plan["needs_resize"] is False


def test_select_overshoots_when_no_exact_factor_exists():
  # There is no x3 model, so the smallest model above three should be chosen and
  # the result resampled down.
  plan = select.select(3)
  assert plan["factor"] == 4
  assert plan["needs_resize"] is True


def test_select_chains_models_to_reach_a_large_factor():
  # x8 cannot be reached by any single registered model, so it must be composed
  # from smaller factors, and the combined factor must cover the target.
  plan = select.select(8)
  assert plan["factor"] >= 8
  assert len(plan["models"]) >= 2


def test_select_plan_reports_resize_only_when_needed():
  assert select.select(2)["needs_resize"] is False
  assert select.select(2.5)["needs_resize"] is True
