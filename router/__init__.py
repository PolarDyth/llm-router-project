"""Complexity-router domain logic.

Stages, in pipeline order (each stage's output is the next stage's input):

  catalogs -> scoring -> model_registry   (tier per canonical model)
  dataset -> pairs                        (model names -> tiers -> training rows)
  contracts -> metrics                    ((prompt, tier) interface and its eval math)

Deliberately re-exporting nothing: `router.devices` and `router.metrics` pull in
torch, and importing `router.model_registry` must not drag the dependency along.
Scripts are thin argument-parsing shells around these modules.
"""
