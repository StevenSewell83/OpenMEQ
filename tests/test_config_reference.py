"""Tests for `docs/config_reference.md` (M11.4 / OpenMEQ-m1f.4).

`scripts/gen_config_reference.py` generates `docs/config_reference.md` from
the live `openmeq.config.Config` pydantic schema plus that script's own
hand-maintained `MEANINGS` prose. These tests make sure the checked-in
`docs/config_reference.md` can never silently drift from either:

- if a config field is added/renamed/removed in `openmeq/config.py` without
  updating `scripts/gen_config_reference.py`'s `MEANINGS`, the generator
  itself raises (see `test_generator_has_no_missing_or_stale_meanings`);
- if `scripts/gen_config_reference.py` is edited (schema or prose) but
  `docs/config_reference.md` isn't regenerated, `test_config_reference_is_up_to_date`
  fails, naming the exact command to fix it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_REFERENCE = REPO_ROOT / "docs" / "config_reference.md"
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gen_config_reference  # noqa: E402


def test_generator_has_no_missing_or_stale_meanings():
    """`build_reference()` itself raises `SystemExit` if `MEANINGS` and the live
    `openmeq.config.Config` schema have drifted apart in either direction --
    calling it here turns that into a normal, named pytest failure."""
    text = gen_config_reference.build_reference()
    assert "config_reference.md" not in text  # sanity: doesn't choke on its own filename
    assert text.startswith("# OpenMEQ configuration reference")


def test_config_reference_is_up_to_date():
    """The checked-in `docs/config_reference.md` must exactly match what the
    generator produces right now, from the same repo checkout's
    `openmeq.config.Config` schema."""
    expected = gen_config_reference.build_reference()
    assert CONFIG_REFERENCE.is_file(), (
        f"{CONFIG_REFERENCE} does not exist -- run "
        "`python scripts/gen_config_reference.py` to create it."
    )
    actual = CONFIG_REFERENCE.read_text()
    assert actual == expected, (
        f"{CONFIG_REFERENCE} is stale relative to openmeq.config.Config and/or "
        "scripts/gen_config_reference.py's own MEANINGS text -- regenerate it with "
        "`python scripts/gen_config_reference.py`."
    )


def test_every_example_yaml_top_level_key_is_a_config_section():
    """Every top-level key in `config/example.yaml` (M0.3's reference config)
    must be a real `openmeq.config.Config` section, and vice versa -- this is
    the "top-level YAML key" half of the acceptance criteria's consistency
    check; `test_config_reference_is_up_to_date` covers every individual
    field, this one specifically covers section-level drift against the
    example config file itself."""
    import yaml

    from openmeq.config import Config

    with open(EXAMPLE_CONFIG) as f:
        raw = yaml.safe_load(f)

    yaml_sections = set(raw.keys())
    schema_sections = set(Config.model_fields.keys())
    assert yaml_sections == schema_sections, (
        f"config/example.yaml top-level sections {yaml_sections} do not match "
        f"openmeq.config.Config's sections {schema_sections}"
    )


def test_every_config_reference_key_documents_a_real_leaf_field():
    """Every `Key` cell in `docs/config_reference.md`'s tables corresponds to a
    real leaf path in the live schema (redundant with the generator's own
    internal check, but asserted independently here directly against the
    checked-in file, in case it was ever hand-edited out of band)."""
    from openmeq.config import Config

    leaves = {leaf.path for leaf in gen_config_reference.walk(Config)}
    documented = set()
    for line in CONFIG_REFERENCE.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        key = line.split("`", 2)[1]
        documented.add(key)

    assert documented == leaves, (
        f"docs/config_reference.md documents {documented - leaves} which are not real "
        f"openmeq.config.Config fields, and is missing {leaves - documented}"
    )
