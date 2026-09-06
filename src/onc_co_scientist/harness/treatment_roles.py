"""Public treatment roles, expressed only in the dataset's visible column names."""

from collections.abc import Mapping, Sequence

TREATMENT_ROLE_VERSION = "explicit-treatment-columns-v1"


def visible_treatment_columns(
    treatment_columns: Sequence[str],
    dataset_columns: Sequence[str],
    mapping: Mapping[str, str] | None = None,
) -> list[str]:
    """Project every treatment role into a public schema; never select by effect."""
    columns = [mapping[name] if mapping is not None else name for name in treatment_columns]
    missing = set(columns) - set(dataset_columns)
    if missing:
        raise ValueError(f"Treatment columns absent from public dataset: {sorted(missing)}")
    if len(set(columns)) != len(columns):
        raise ValueError("Treatment columns must be unique")
    return columns


def render_treatment_roles(columns: Sequence[str]) -> str:
    """Shared wording for task bundles and every workflow participant."""
    if not columns:
        return ""
    return (
        "## Treatment variables\n\n"
        "The following columns are treatment variables (treatment/exposure indicators):\n\n"
        + "\n".join(f"- `{column}`" for column in columns)
        + "\n\nFor a treatment-effect or treatment-interaction hypothesis, use the treatment "
        "being tested as the exposure and describe the subgroup separately using its "
        "modifier/covariate columns. These roles apply regardless of whether column names "
        "are descriptive or masked. The list identifies all treatment variables; it does "
        "not imply that any particular treatment has an effect.\n"
    )
