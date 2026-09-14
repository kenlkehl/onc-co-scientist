"""Compose the clinical context, result tables, and three vector figure collections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

QUESTIONS = {
    "nlr_ge_3": "Neutrophil-to-lymphocyte ratio (NLR) >=3 versus <3",
    "stage_iv": "Stage IV versus other stages",
    "crp_mg_l_ge_10": "C-reactive protein (CRP) >=10 versus <10 mg/L",
    "ecog_ps_ge_2": "ECOG performance status >=2 versus <2",
    "brca2_mutation": "BRCA2 mutation versus no mutation",
    "research_marker_d_positive": "Research marker D positive versus negative",
}
VARIABLES = {
    "research_signature_a": "A", "research_signature_b": "B", "research_signature_c": "C",
}


COVARIATE_GROUPS = {
    "Demographics and disease (7)": {
        "age_years": "age", "sex_female": "sex", "smoking_status": "smoking status",
        "ecog_ps": "ECOG performance status", "histology": "histology",
        "stage_iv": "stage IV indicator", "has_brain_mets": "brain metastases",
    },
    "Molecular and immune markers (7)": {
        "egfr_mutation": "EGFR mutation", "kras_g12c": "KRAS G12C", "alk_fusion": "ALK fusion",
        "stk11_mutation": "STK11 mutation", "brca2_mutation": "BRCA2 mutation",
        "pdl1_tps": "PD-L1 tumor proportion score", "tmb_high": "high tumor mutational burden",
    },
    "Laboratory and nutritional measures (15)": {
        "albumin_g_dl": "albumin", "ldh_u_l": "LDH", "weight_loss_pct_6mo": "6-month weight loss",
        "crp_mg_l": "CRP", "nlr": "NLR", "hemoglobin_g_dl": "hemoglobin",
        "alkaline_phosphatase_u_l": "alkaline phosphatase", "ast_u_l": "AST", "alt_u_l": "ALT",
        "total_bilirubin_mg_dl": "total bilirubin", "creatinine_mg_dl": "creatinine",
        "bun_mg_dl": "blood urea nitrogen", "sodium_meq_l": "sodium",
        "potassium_meq_l": "potassium", "calcium_mg_dl": "calcium",
    },
    "Treatment indicators (4)": {
        "treatment_pembrolizumab": "pembrolizumab", "treatment_sotorasib": "sotorasib",
        "treatment_olaparib": "olaparib", "treatment_osimertinib": "osimertinib",
    },
    "Constructed research variables (6)": {
        "research_signature_a": "signature A", "research_signature_b": "signature B",
        "research_signature_c": "signature C", "research_marker_d_positive": "marker D",
        "research_marker_e_positive": "marker E", "research_marker_f_positive": "marker F",
    },
    "Derived threshold indicators (4)": {
        "ecog_ps_ge_2": "ECOG >=2", "crp_mg_l_ge_10": "CRP >=10 mg/L",
        "albumin_g_dl_le_3_5": "albumin <=3.5 g/dL", "nlr_ge_3": "NLR >=3",
    },
}


def clinical_context(repo, config):
    root = repo / config["clinical_context_root"]
    paths = [root / n for n in ("pair.json", "assignment.json", "expected_equation.txt",
                               "surprising_equation.txt")]
    paths.append(repo / config["clinical_masking_file"])
    assignments = json.loads((root / "assignment.json").read_text())
    dictionary_path = root.parent.parent / "public" / assignments["expected"]["task_id"] / "data_dictionary.json"
    paths.append(dictionary_path)
    raw = {p.name: p.read_bytes() for p in paths}
    pair = json.loads(raw["pair.json"])
    if pair["focal_id"] != "nsclc_nlr_ge_3_pfs":
        raise ValueError("Update the clinical narrative for a different focal discovery")
    mapping = json.loads(raw["masking.json"])["columns"]
    dictionary = json.loads(raw["data_dictionary.json"])
    covariates = [k for fields in COVARIATE_GROUPS.values() for k in fields]
    if len(covariates) != len(set(covariates)) or set(covariates) != set(mapping):
        raise ValueError("Update the covariate overview to match the frozen masking map")
    if set(dictionary) != set(covariates) | {"patient_id", "log_pfs_months"}:
        raise ValueError("Frozen dataset dictionary differs from the overview inventory")
    equations = {}
    for condition in ("expected", "surprising"):
        equations[condition] = dict(line.split(" = ", 1) for line in
                                    raw[condition + "_equation.txt"].decode().splitlines()
                                    if line.startswith("nsclc_"))
    ids = {d["hypothesis"]["id"] for d in pair["discoveries"]}
    if set(equations["expected"]) != ids or set(equations["surprising"]) != ids:
        raise ValueError("Clinical equations do not match the discovery inventory")
    changed = [i for i in ids if equations["expected"][i] != equations["surprising"][i]]
    if changed != [pair["focal_id"]]:
        raise ValueError("Expected exactly the focal discovery to change between equations")
    discoveries = []
    for d in pair["discoveries"]:
        h = d["hypothesis"]
        if h["outcome"] != "log_pfs_months" or h["contrast"] != "mean_difference":
            raise ValueError("Clinical report requires the recorded log-PFS mean contrasts")
        if (h["exposed"], h["comparator"]) != (1, 0):
            raise ValueError("Clinical wording requires the recorded binary exposure coding")
        focal = h["id"] == pair["focal_id"]
        question = QUESTIONS[h["exposure"]]
        subgroup = " and ".join(
            f"{VARIABLES.get(s['variable'], s['variable'])} "
            f"{ {'ge': '>=', 'le': '<=', 'eq': '='}[s['op']]} {s['value']:g}"
            for s in h["subgroup"]
        )
        if h["eligibility"]:
            raise ValueError("Add clinical wording for nonempty eligibility restrictions")
        if subgroup:
            question += ", within " + subgroup
        coefficients = {}
        for condition in equations:
            match = re.search(r"\*\((-?\d+(?:\.\d+)?)\*I", equations[condition][h["id"]])
            if not match:
                raise ValueError("Unrecognized clinical equation")
            coefficients[condition] = float(match[1])
        if focal and coefficients["expected"] != -coefficients["surprising"]:
            raise ValueError("Focal coefficient must reverse sign")
        discoveries.append(dict(
            id=h["id"], question=question, exposure=h["exposure"],
            masked_exposure=mapping[h["exposure"]], focal=focal,
            expected_category=d["category"],
            surprising_category="surprising" if focal else d["category"],
            coefficients=coefficients,
        ))
    focal = next(d for d in discoveries if d["focal"])
    if focal["coefficients"] != {"expected": -0.45, "surprising": 0.45}:
        raise ValueError("Update the clinical narrative for changed focal coefficients")
    return dict(pair_id=pair["pair_id"], n=pair["n"], focal_id=pair["focal_id"],
                discoveries=discoveries, assignments=assignments,
                covariate_count=len(covariates), covariate_groups=COVARIATE_GROUPS,
                data_dictionary=dictionary,
                masking=mapping, sources=[dict(path=str(p.relative_to(repo)),
                    sha256=hashlib.sha256(raw[p.name]).hexdigest()) for p in paths])


def build(out, repo, config):
    import importlib.util
    from pypdf import PdfReader, PdfWriter, Transformation
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Table, TableStyle

    fonts = Path(importlib.util.find_spec("matplotlib").origin).parent / "mpl-data/fonts/ttf"
    pdfmetrics.registerFont(TTFont("ReportSans", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSansBold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold")

    metadata = json.loads((out / "provenance.json").read_text())
    if not metadata.get("figure_generation"):
        raise ValueError("Tables were refreshed without figures; run --figures-only first")
    with (out / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with (out / "cross_condition_scores.csv").open() as handle:
        cross = list(csv.DictReader(handle))
    with (out / "run_metrics.csv").open() as handle:
        runs = list(csv.DictReader(handle))
    from build_presentation_results import comparison_order, key, persistent_comparison
    rows.sort(key=lambda row: comparison_order(key(row)))
    cross.sort(key=lambda row: comparison_order(key(row)))
    focused_count = len({key(row) for row in rows if persistent_comparison(key(row))})
    costs = json.loads((out / "cost_estimates.json").read_text())
    if costs["cohort_sha256"] != hashlib.sha256((out / "run_metrics.csv").read_bytes()).hexdigest():
        raise ValueError("Cost estimates differ from the report run inventory; rebuild costs first")
    context = clinical_context(repo, config)
    if {r["pair_id"] for r in runs if r["pair_id"]} != {context["pair_id"]}:
        raise ValueError("Result dataset pair differs from the clinical context")
    context_path = out / "clinical_context.json"
    context_path.write_text(json.dumps(context, indent=2) + "\n")
    names = {"gpt-6-astra": "Astra", "gpt-5.6-sol": "Sol", "gpt-5.6-terra": "Terra",
             "gpt-5.6-luna": "Luna", "claude-opus-5": "Claude Opus 5",
             "Inferact/Qwen3.8-27B-NVFP4": "Qwen 3.8 27B",
             "RedHatAI/Gemma-4-31B-IT-FP8-Dynamic": "Gemma 4 31B"}
    metrics = list(dict.fromkeys(r["metric"] for r in rows))
    conditions = ("expected-unmasked", "expected-masked",
                  "surprising-unmasked", "surprising-masked")
    width, height, margin = 1152, 648, 43
    ink, muted, blue = colors.HexColor("#172334"), colors.HexColor("#526074"), colors.HexColor("#2575ad")
    style = ParagraphStyle("body", fontName="ReportSans", fontSize=13, leading=19,
                           textColor=ink, spaceAfter=12)
    small = ParagraphStyle("small", parent=style, fontSize=10, leading=14)
    cell = ParagraphStyle("cell", parent=style, fontSize=10, leading=13)
    tiny = ParagraphStyle("tiny", parent=style, fontSize=8, leading=10)

    with tempfile.TemporaryDirectory(prefix="clinical-report-") as temp:
        narrative = Path(temp) / "tables.pdf"
        c = canvas.Canvas(str(narrative), pagesize=(width, height))
        c.setTitle("Clinical experiment report")
        def heading(title, subtitle=""):
            c.setFillColor(ink)
            c.setFont("ReportSansBold", 27)
            c.drawString(margin, height - 59, title)
            if subtitle:
                c.setFont("ReportSans", 12)
                c.setFillColor(muted)
                c.drawString(margin, height - 82, subtitle)

        def paragraph(text, y, paragraph_style=style, x=margin, available=None):
            p = Paragraph(text, paragraph_style)
            _, h = p.wrap(available or width - 2 * margin, height)
            if y - h < 38:
                raise ValueError("PDF paragraph exceeds the page; revise layout")
            p.drawOn(c, x, y - h)
            return y - h - paragraph_style.spaceAfter

        def draw_table(data, widths, y, font=cell, focal_row=False, padding=5,
                       header_rows=1, spans=(), section_break=None):
            content = [[Paragraph(str(value), font) for value in row] for row in data]
            t = Table(content, colWidths=widths, repeatRows=header_rows)
            commands = [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e8eff6")),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
                 [colors.white, colors.HexColor("#f5f8fb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), .8, blue),
            ]
            commands.extend(("SPAN", start, end) for start, end in spans)
            if section_break is not None and section_break < len(data):
                commands.append(("LINEABOVE", (0, section_break), (-1, section_break), 1.3, blue))
            if focal_row:
                commands.append(("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e5f2f6")))
            t.setStyle(TableStyle(commands))
            _, h = t.wrap(width - 2 * margin, height)
            if y - h < 45:
                raise ValueError(f"PDF table is too tall ({h:.0f} points)")
            t.drawOn(c, margin, y - h)
            return y - h - 16

        heading("Clinical experiment report", "Dataset overview | synthetic non-small cell lung cancer (NSCLC)")
        overview_style = ParagraphStyle("overview", parent=small, fontSize=11, leading=16)
        y = paragraph(f"<b>N = {context['n']:,} simulated patients per dataset version.</b> "
                      f"There are <b>{context['covariate_count']} covariate columns</b>, including derived "
                      "indicators, plus a patient identifier and one outcome: natural-log "
                      "progression-free survival (PFS) in months, fully observed without censoring.",
                      538, overview_style)
        y = paragraph("<b>Key finding deliberately flipped:</b> NLR >=3 versus &lt;3 is associated "
                      "with <b>shorter PFS in the expected version</b> and <b>longer PFS in the surprising "
                      "version</b>. Its injected log-PFS coefficient changes from <b>-0.45 to +0.45</b>. "
                      "The paired versions share covariates and random residuals; only this outcome term changes.",
                      y, overview_style)
        covariate_table = [["<b>Covariate group</b>", "<b>Included variables</b>"]]
        for label, fields in context["covariate_groups"].items():
            covariate_table.append([escape(label), escape("; ".join(fields.values()))])
        y = draw_table(covariate_table, [267, 799], y, font=small, padding=3)
        findings = [
            ["<b>Planted finding category</b>", "<b>Associations encoded in the synthetic data</b>"],
            ["<b>Expected</b>", "Stage IV and CRP >=10 mg/L each predict shorter PFS in both versions. "
             "NLR >=3 also predicts shorter PFS in the expected version."],
            ["<b>Surprising</b>", "ECOG >=2 predicts longer PFS in both versions. In the surprising "
             "version, NLR >=3 also predicts longer PFS."],
            ["<b>Neutral</b>", "BRCA2 mutation predicts longer PFS within A >=0 and B >=0.5; "
             "marker D positivity predicts longer PFS within A >=0 and C <=0.5, in both versions. "
             "Neutral describes the prior-knowledge category, not a null effect."],
        ]
        y = draw_table(findings, [267, 799], y, font=small, padding=3)
        paragraph("<b>Four versions:</b> expected-unmasked, expected-masked, surprising-unmasked, "
                  "surprising-masked. Masking changes names and categorical text labels while preserving "
                  "the numerical data. Each version has six planted findings: <b>3 expected / 2 neutral / "
                  "1 surprising</b> before the focal flip, versus <b>2 / 2 / 2</b> after it. "
                  "These categories follow the frozen benchmark specification. Full comparisons are on page 3.",
                  y, small)
        c.showPage()

        # Reader's guide, including the saved results' cohort rather than today's state.
        heading("Results snapshot and contents", "NSCLC discovery benchmark | tables and figures with 95% confidence intervals")
        counts = metadata["counts"]
        y = paragraph(f"<b>Results snapshot:</b> {escape(metadata['snapshot_started_at'])}<br/>"
                      f"<b>{metadata['runs']} selected run slots</b>; "
                      f"{counts.get('completed', 0)} successfully completed; "
                      f"{counts.get('failed', 0)} ended with errors; "
                      f"{counts.get('unfinished', 0) + counts.get('queued', 0)} unfinished or queued.", 525)
        focal = next(d for d in context["discoveries"] if d["focal"])
        y = paragraph("<b>The manipulated discovery:</b> " + escape(focal["question"]) +
                      ". Its simulated relationship with progression-free survival (PFS) "
                      "changes from <b>shorter PFS (expected)</b> to <b>longer PFS (surprising)</b>. "
                      "The other five discovery definitions remain fixed.", y - 7)
        y = paragraph("<b>Four experimental conditions:</b> expected-unmasked, expected-masked, "
                      "surprising-unmasked, and surprising-masked. Expected/surprising describes "
                      "the focal finding's direction. Masked/unmasked describes the variable labels "
                      "visible to the agent; masking preserves the numerical data.", y)
        table_start = 9
        cost_start = table_start + len(metrics) + 1
        figure_start = cost_start + 2
        persistent = PdfReader(out / "figures/persistent_only/all_metrics.pdf")
        prior = PdfReader(out / "figures/all_metrics.pdf")
        grouped = PdfReader(out / "figures/by_model_harness/all_model_harnesses.pdf")
        prior_start = figure_start + len(persistent.pages)
        group_start = prior_start + len(prior.pages)
        entries = [("Dataset overview: N, covariates, findings and focal reversal", "1"),
                   ("Clinical questions and planted discoveries", "3"),
                   ("Dataset construction: application schematic", "4"),
                   ("Custom runner: flowchart of one run", "5"),
                   ("Reading the results: individual metrics, uncertainty, summary and interaction", "6-8"),
                   ("Results tables: condition metrics, summary and separate interaction", f"{table_start}-{cost_start - 1}"),
                   ("API and GPU electricity costs: estimates and assumptions", f"{cost_start}-{figure_start - 1}"),
                   ("Persistent + Biomni figures (one page per metric)", f"{figure_start}-{prior_start - 1}"),
                   ("All-workflow figures (one page per metric)", f"{prior_start}-{group_start - 1}"),
                   ("Within-model/harness figures", f"{group_start}-{group_start + len(grouped.pages) - 1}"),
                   ("Source files and model identifiers", str(group_start + len(grouped.pages)))]
        y = draw_table([["<b>Contents</b>", "<b>Pages</b>"]] + entries, [890, 176], y - 8)
        paragraph("All findings described here are deliberately embedded in synthetic data. "
                  "The categories reflect the benchmark's frozen specification.", y, small)
        c.showPage()

        heading("Clinical questions and planted discoveries", f"{context['n']:,} simulated NSCLC patients per version | six planted comparisons")
        y = paragraph("The question for each comparison is whether progression-free survival differs "
                      "between the exposed and comparator groups. The endpoint is mean natural-log "
                      "PFS in months, fully observed without censoring. A, B, and C are constructed "
                      "research signatures in arbitrary units; marker D is a constructed binary assay.", 538)
        inventory = [["<b>Underlying comparison</b>", "<b>Masked exposure<br/>(main grid)</b>",
                      "<b>Expected-focal dataset</b>", "<b>Surprising-focal dataset</b>"]]
        for d in context["discoveries"]:
            text = escape(d["question"])
            if d["focal"]:
                text = "<b>FLIPPED: " + text + "</b>"
            values = []
            for condition in ("expected", "surprising"):
                coefficient = d["coefficients"][condition]
                direction = "Shorter" if coefficient < 0 else "Longer"
                values.append(f"{direction} PFS; {d[condition + '_category']}<br/>"
                              f"Injected coefficient: {coefficient:+.2f}")
            inventory.append([text, d["masked_exposure"], *values])
        y = draw_table(inventory, [425, 125, 258, 258], y, focal_row=True)
        y = paragraph("<b>What changes:</b> the NLR >=3 term changes from -0.45 to +0.45 log-PFS "
                      "units. The paired datasets share covariates and random residuals; only this "
                      "outcome contribution changes. These coefficients are injected model terms, "
                      "not hazard ratios or estimated marginal contrasts.", y, small)
        y = paragraph("<b>What stays surprising:</b> ECOG >=2 predicts longer PFS in both versions. "
                      "The inventory therefore changes from 3 expected / 2 neutral / 1 surprising "
                      "to 2 expected / 2 neutral / 2 surprising findings. Both dataset versions "
                      "contain a mixture of discovery categories.", y, small)
        masking = context["masking"]
        paragraph(f"<b>Main-grid masked identities:</b> NLR >=3 = {masking['nlr_ge_3']}; "
                  f"A = {masking['research_signature_a']}; B = {masking['research_signature_b']}; "
                  f"C = {masking['research_signature_c']}. Numerical subgroup thresholds and outcomes "
                  "are preserved. Source: the frozen pair, equations, and masking map "
                  "listed in the source appendix and clinical_context.json.", y, small)
        c.showPage()

        from presentation_workflow import draw_custom_workflow, draw_dataset_construction

        draw_dataset_construction(c, out)
        c.showPage()
        draw_custom_workflow(c, out)
        c.showPage()

        heading("Reading the results", "What each metric measures and how it is calculated")
        cohort = ("successfully completed runs only" if metadata["cohort"] == "completed" else
                  "all ended runs with reports, including runs that ended with errors")
        y = paragraph("<b>Multiple findings per dataset:</b> each version contains six planted findings "
                      "(listed on page 3), and a run can propose additional findings. Precision and recall "
                      "count findings within each run. The reported values then average run scores within "
                      "each model/harness/workflow and experimental condition; individual patients are "
                      "not the scoring units.", 538)
        definitions = [
            ("Precision (%)", "<b>Of the findings the agent accepts at the end, how many hold up?</b> "
             "100 x (distinct accepted claims that were tested and supported on fresh confirmation data) "
             "/ (all distinct accepted claims). Example: 3 confirmed out of 4 accepted = 75%. Additional "
             "findings outside the six planted targets can count. Duplicate claims count once; no accepted "
             "claims makes precision unavailable."),
            ("Recall (%)", "<b>How many of the planted findings did the agent recover?</b> Within each "
             "expected, neutral, and surprising finding category, divide recovered targets by available "
             "targets; average those three fractions and multiply by 100. Recovery requires testing, "
             "final acceptance, fresh-data confirmation, and an exact match including subgroup and direction. "
             "Each target counts once. In the expected-focal dataset, recovering only its 3 expected findings "
             "gives (3/3 + 0/2 + 0/1) / 3 x 100 = 33.3%."),
            ("Scientific F1*", "<b>Combines precision and recall:</b> 2 x precision x recall / "
             "(precision + recall), calculated per run and then averaged. Undefined precision or a zero "
             "denominator gives F1 = 0 under the evaluator's convention."),
            ("Exploration E", "<b>How broadly and how early did the agent investigate the planted "
             "comparisons?</b> At the end of each of 25 rounds, calculate the fraction of each finding "
             "category tested so far. Average across the three categories and all 25 rounds, then multiply "
             "by 100. A valid test of the exact comparison counts regardless of the proposed direction, "
             "test result, or later acceptance. Repeating a test adds no coverage. Earlier tests contribute "
             "to more rounds and therefore earn more credit. Biomni uses 25 reporting rounds."),
            ("Evidence responsiveness B", "<b>Does the agent's judgment follow the validation evidence?</b> "
             "Score its recorded decision at the scheduled check two rounds after each eligible validation result: accept when "
             "the result's whole interval is above the minimum-effect cutoff; reject when wholly below; "
             "leave unresolved when the interval spans or touches the cutoff. These are the supported, "
             "excluded, and ambiguous evidence classes. Within a run and class, accuracy = correct "
             "decisions / eligible results. Average each class's accuracy across available runs, then "
             "average the three class means equally and multiply by 100."),
            ("Focal recovery (%)", "<b>Was the flipped NLR/PFS finding recovered?</b> Each run scores "
             "100 for an exact, tested, accepted, independently confirmed recovery, or 0 otherwise. "
             "The displayed mean is the percentage of runs recovering this one finding."),
        ]
        y = draw_table([["<b>Metric (all on a 0-100 scale)</b>", "<b>Meaning and calculation</b>"]] + definitions,
                       [220, 846], y, font=small, padding=6)
        paragraph("Finding categories describe the six targets. Expected/surprising column labels "
                  "describe which direction was assigned to the focal NLR finding. These are different "
                  "uses of the same terms. Scoring and aggregation details continue on page 7.", y, small)
        c.showPage()

        heading("Reading the results: scoring details", "Eligible observations, uncertainty, and execution accounting")
        y = paragraph(f"<b>Scientific cohort:</b> {cohort}. Unfinished and queued runs never receive "
                      "zero scores. Means are calculated separately for each of the four conditions. "
                      "Runs within a base dataset receive equal weight; base datasets then receive equal "
                      "weight. This report has one base dataset pair. Undefined observations are omitted. "
                      "Scientific F1 and focal recovery use the committed scientific results without the "
                      "legacy whole-run zero penalty for execution errors.", 538, small)
        y = paragraph("<b>Confirmation and effect thresholds:</b> after a run, the evaluator tests its "
                      "final accepted claims on fresh simulated data. Confirmation requires the interval's "
                      "lower bound to exceed the benchmark's minimum effect in the claimed direction, "
                      "with a multiplicity adjustment across final accepted claims. The NSCLC cutoff is "
                      "0.10 natural-log PFS units. An excluded result rules out reaching this cutoff; it "
                      "need not imply an exactly zero or opposite effect. F1* combines category-balanced "
                      "target recall with a precision measure that can credit additional findings.", y, small)
        y = paragraph("<b>Evidence responsiveness eligibility:</b> use voluntary and automatically "
                      "delivered validation results. Invalid results and results whose two-round "
                      "follow-up falls beyond the budget or reached run history are excluded. A missing "
                      "decision at an eligible follow-up scores zero. If a run lacks an evidence class, "
                      "its accuracy for that class is unavailable. Other runs can supply that class; "
                      "the aggregate B is unavailable only when a class is absent across all eligible runs. "
                      "Each class receives equal weight even if the numbers of results differ.", y, small)
        y = paragraph("<b>95% CIs:</b> 10,000 seeded whole-run bootstrap draws; recompute the "
                      "class-balanced B statistic in each draw. Focal recovery uses Wilson intervals. "
                      "CIs are unavailable with too few eligible runs or evidence classes. Constant "
                      "continuous scores can have zero-width bootstrap intervals. These are marginal "
                      "intervals, not paired effect tests or multiplicity-adjusted intervals. They describe "
                      "repeated-run uncertainty on this fixed dataset, not uncertainty across clinical "
                      "datasets, and are distinct from the validation-result intervals used to score B.", y, small)
        y = paragraph("<b>Counts and tokens:</b> attempted counts selected run identities that started; "
                      "successfully completed means all stages succeeded; errors means status failed. "
                      "Counts span all four conditions. Median output tokens use successful runs only, "
                      "include returned retries and peer/chair calls, and can be lower bounds when usage "
                      "is missing. Per-condition denominators and usage coverage are in the companion CSVs.", y, small)
        paragraph("<b>Workflows:</b> persistent retains bounded conversation and the ledger; sequential "
                  "starts fresh at each stage with the ledger; deliberative uses peer drafts and a chair. "
                  "Codex denotes CLI transport within this project's scientific runner. Biomni uses its "
                  "own workflow with full resources and xhigh reasoning; other runs use medium. "
                  "Mixed code generations and budgets remain important comparison limitations.", y, small)
        c.showPage()

        heading("Reading the results: summary and interaction", "Overall performance and the label-by-surprise contrast answer different questions")
        y = paragraph("<b>Weighted condition score C (0-100):</b> within each model/harness/workflow "
                      "and condition, combine the aggregate metric values: <b>C = 0.35 x F1 + "
                      "0.25 x focal recovery + 0.20 x exploration + 0.20 x responsiveness.</b> "
                      "Use the reported mean of per-run F1, not F1 recalculated from mean precision "
                      "and recall. B retains its evidence-class balancing. Precision and recall enter "
                      "through F1; extra focal weight deliberately emphasizes the flipped finding.", 538, small)
        y = paragraph("<b>Grounded discovery and adaptation S (0-100; higher is better):</b> "
                      "<b>S = (C_EU x C_EM x C_SU x C_SM)<super>1/4</super>.</b> EU = expected-unmasked; "
                      "EM = expected-masked; SU = surprising-unmasked; SM = surprising-masked. "
                      "All four conditions have equal weight. This geometric mean makes weakness in "
                      "one condition harder to offset elsewhere. A true zero C gives S = 0; a missing "
                      "required component or condition makes S unavailable, without reweighting.", y, small)
        y = paragraph("<b>Separate label-by-surprise interaction I (percentage points):</b> "
                      "<b>I = (focal_EU - focal_SU) - (focal_EM - focal_SM).</b> Positive means the "
                      "expected-to-surprising recovery drop is larger with recognizable labels; "
                      "negative means it is larger with masked labels. Example: an unmasked drop of "
                      "40 points minus a masked drop of 10 points gives I = +30. Zero can reflect "
                      "uniformly good or uniformly poor recovery, so read I alongside S and the focal "
                      "condition table. I is not subtracted from S and is not a higher-is-better score.", y, small)
        y = paragraph("<b>95% CIs for C and S:</b> 10,000 seeded bootstrap samples of whole runs, "
                      "drawn independently within each condition. Recompute all component means, B's "
                      "class balance, C, and then S in each draw, preserving within-run metric relationships. "
                      "Do not average CI endpoints or pair runs merely by replicate number. At least "
                      "two observations per required component/class and 95% valid draws are required. "
                      "Sparse evidence can leave a point estimate available but its CI unavailable; "
                      "constant samples can yield zero-width bootstrap CIs.", y, small)
        y = paragraph("<b>95% CI for I:</b> Wilson-based method of variance estimates recovery "
                      "(MOVER), combining the four independent focal-recovery proportions. This avoids "
                      "a collapsed interval when a condition has all successes or all failures. "
                      "The intervals are approximate, conditional on this fixed clinical dataset pair, "
                      "and not adjusted for multiple comparisons. Method: "
                      '<link href="https://doi.org/10.1016/j.csda.2008.09.033" color="#2575ad">'
                      "Zou, Huang and Zhang (2009), Computational Statistics &amp; Data Analysis 53:1080-1085.</link>", y, small)
        paragraph("<b>Interpretation limits:</b> S is a provisional descriptive index with weights "
                  "chosen after these experiments, not a validated or preregistered endpoint. I does "
                  "not establish pretraining bias: masking also changes interpretability. Focal recovery "
                  "avoids changing category weights when the target flips. Exploration covers the six "
                  "planted comparisons; B includes automatic validation and already-correct judgments. "
                  "Neither directly measures keeping a claim open or appropriately choosing validation.", y, small)
        c.showPage()

        for metric in metrics:
            heading(metric, f"Persistent + Biomni: first {focused_count} rows | Other workflows below the divider | Means and 95% CIs")
            headers = ["LLM", "Harness", "Workflow", "Effort", "Attempted<br/>runs",
                       "Completed<br/>runs", "Failed<br/>runs", "Median output<br/>tokens / success",
                       "Expected<br/>unmasked", "", "Expected<br/>masked", "",
                       "Surprising<br/>unmasked", "", "Surprising<br/>masked", ""]
            data = [[f"<b>{h}</b>" for h in headers],
                    [""] * 8 + ["<b>Mean</b>", "<b>95% CI</b>"] * 4]
            for row in rows:
                if row["metric"] != metric:
                    continue
                median = row["median_output_tokens_per_completed_run"]
                values = []
                for condition in conditions:
                    low, high = row[condition + "_ci95_low"], row[condition + "_ci95_high"]
                    values.extend([f"{float(row[condition]):.1f}" if row[condition] else "NA",
                                   f"{float(low):.1f}-{float(high):.1f}" if low and high else "NA"])
                data.append([names.get(row["llm"], row["llm"]), row["harness"], row["workflow"],
                             row["reasoning_effort"], row["runs_attempted"],
                             row["runs_successfully_completed"], row["runs_ended_with_errors"],
                             (f"{float(median):,.0f}" if float(median).is_integer()
                              else f"{float(median):,.1f}") if median else "NA",
                             *values])
            spans = [((i, 0), (i, 1)) for i in range(8)]
            spans.extend(((i, 0), (i + 1, 0)) for i in range(8, 16, 2))
            y = draw_table(data, [107, 91, 79, 50, 72, 72, 55, 100] + [40, 70] * 4, 538,
                           font=ParagraphStyle("results", parent=cell, fontSize=8.2, leading=10.5),
                           padding=3, header_rows=2, spans=spans, section_break=2 + focused_count)
            paragraph(f"Cohort: {cohort}. Counts and token medians span all four conditions; "
                      "condition means use available scored runs. NA = unavailable, not zero. "
                      "95% CIs: whole-run bootstrap; Wilson for focal recovery. NA in a CI column "
                      "means insufficient eligible runs or evidence. Exact model identifiers are in the appendix. "
                      "Full-precision values and "
                      "per-condition counts: results.csv and condition_metrics.csv.", y, small)
            c.showPage()
        heading("Summary and separate focal interaction", f"Persistent + Biomni: first {focused_count} rows | Other workflows below the divider | 95% confidence intervals")
        cross_keys = ("llm", "harness", "workflow", "reasoning_effort")
        groups = list(dict.fromkeys(tuple(r[k] for k in cross_keys) for r in cross))
        lookup = {(tuple(r[k] for k in cross_keys), r["metric"]): r for r in cross}
        headers = ["LLM", "Harness", "Workflow", "Effort", "Attempted<br/>runs", "Completed<br/>runs",
                   "Failed<br/>runs", "Median output<br/>tokens / success", "Summary S<br/>(0-100)",
                   "95% CI", "Interaction I<br/>(pp)", "95% CI"]
        data = [["<b>" + h + "</b>" for h in headers]]
        for group in groups:
            row = lookup[group, "summary_score"]
            median = row["median_output_tokens_per_completed_run"]
            values = []
            for metric in ("summary_score", "label_surprise_interaction"):
                item = lookup[group, metric]
                point, low, high = (item[k] for k in ("value", "ci95_low", "ci95_high"))
                signed = "+" if metric == "label_surprise_interaction" else ""
                values.extend([format(float(point), signed + ".1f") if point else "NA",
                               f"{float(low):.1f} to {float(high):.1f}" if low and high else "NA"])
            data.append([names.get(row["llm"], row["llm"]), row["harness"], row["workflow"],
                         row["reasoning_effort"], row["runs_attempted"], row["runs_successfully_completed"],
                         row["runs_ended_with_errors"], f"{float(median):,.0f}" if median else "NA", *values])
        y = draw_table(data, [107, 91, 79, 50, 72, 72, 55, 100, 95, 125, 95, 125], 538,
                       font=ParagraphStyle("cross_results", parent=cell, fontSize=8.2, leading=10.5), padding=3, section_break=1 + focused_count)
        y = paragraph("<b>S:</b> equal-weight geometric mean of the four weighted condition scores; "
                      "higher is better. <b>I:</b> (focal EU - SU) - (focal EM - SM), in percentage points; "
                      "positive means a larger surprise penalty with recognizable labels. I is a signed "
                      "contrast, not a performance score. See page 8 for formulas and limitations.", y, small)
        paragraph("95% CIs: joint whole-run bootstrap for S; Wilson-based MOVER for I. NA = unavailable, "
                  "not zero. Missing CIs indicate insufficient eligible runs/evidence. Counts and tokens "
                  "span all four conditions. Full precision and CI availability reasons: cross_condition_scores.csv.", y, small)
        c.showPage()
        heading("Estimated run costs", "Standard API-equivalent charges and modeled electricity on existing GPUs | USD")
        e = costs["assumptions"]["electricity"]
        y = paragraph("<b>Totals include known usage from all attempted runs, including failures.</b> "
                      "API prices include recorded cache discounts; GPU estimates cover active electricity only. "
                      "Input tokens include cached input. M = million tokens.", 538, small)
        data = [["<b>Model / harness</b>", "<b>Cost basis</b>", "<b>Attempted /<br/>completed</b>",
                 "<b>Input, M</b>", "<b>Output, M</b>", "<b>Total cost</b>"]]
        def money(value):
            return f"${value:,.2f}" if value is not None else "NA"
        for row in costs["totals"]:
            data.append([names.get(row["llm"], row["llm"]) + " / " + row["harness"],
                         row["cost_basis"], f"{row['runs_attempted']} / {row['runs_completed']}",
                         f"{row['input_tokens'] / 1e6:,.2f}", f"{row['output_tokens'] / 1e6:,.2f}",
                         money(row["total_known_cost_usd"])])
        y = draw_table(data, [295, 170, 145, 145, 145, 166], y, font=small, padding=3)
        y = paragraph("<b>Mean recorded cost per completed run</b>, averaged across all four dataset conditions. "
                      "Failed runs are excluded from these means. Dash = workflow not used.", y - 6, small)
        data = [["<b>Model / harness</b>", "<b>Persistent</b>", "<b>Sequential</b>",
                 "<b>Deliberative</b>", "<b>Biomni</b>"]]
        for row in costs["totals"]:
            vals = []
            for workflow in ("persistent", "sequential", "deliberative", "Biomni"):
                match = next((r for r in costs["workflows"] if r["llm"] == row["llm"]
                              and r["harness"] == row["harness"] and r["workflow"] == workflow), None)
                vals.append(money(match["mean_cost_per_completed_run_usd"]) if match else "-")
            data.append([names.get(row["llm"], row["llm"]) + " / " + row["harness"], *vals])
        y = draw_table(data, [346, 180, 180, 180, 180], y, font=small, padding=3)
        paragraph(f"Electricity: ${e['usd_per_kwh']:g}/kWh; {e['gpu_count']:g} GPU at {e['gpu_watts']:g} W; "
                  f"{e['input_tokens_per_second']:g} input and {e['output_tokens_per_second']:g} output tokens/s. "
                  "These electricity costs are not full serving costs. Calculation and limitations follow.", y - 5, small)
        c.showPage()

        heading("How costs are estimated", "Reproducible token accounting | explicit rate and hardware assumptions")
        assumptions = costs["assumptions"]
        y = paragraph(f"<b>API pricing snapshot: {assumptions['pricing_verified_on']}.</b> Standard global rates "
                      "in USD per million tokens, without fast, batch, flex, or regional adjustments. "
                      "Prices are frozen in presentation_costs.json; update that file when rates change.", 538, small)
        data = [["<b>Model</b>", "<b>Input</b>", "<b>Cached input</b>", "<b>Cache write</b>", "<b>Output</b>"]]
        for model, rates in assumptions["api_usd_per_million_tokens"].items():
            data.append([names.get(model, model)] + [money(rates[k]) for k in
                        ("input", "cached_input", "cache_write", "output")])
        y = draw_table(data, [346, 180, 180, 180, 180], y, font=small, padding=3)
        y = paragraph("<b>API formula:</b> [(input - cached - cache-write tokens) x input rate + "
                      "cached tokens x cached rate + cache-write tokens x write rate + output tokens x output rate] / 1,000,000. "
                      "Output already includes reasoning tokens. OpenAI requests above 272,000 input tokens use "
                      "2x input/cache rates and 1.5x output rates. Claude's adapter does not request prompt caching; "
                      "its recorded input is priced as uncached.", y, small)
        y = paragraph(f"<b>Electricity formula:</b> hours = (input tokens / {e['input_tokens_per_second']:g} + "
                      f"output tokens / {e['output_tokens_per_second']:g}) / 3,600; "
                      f"kWh = hours x {e['gpu_watts']/1000:g} kW x {e['gpu_count']:g} GPU; "
                      f"cost = kWh x ${e['usd_per_kwh']:g}. Prefill and generation times are added. "
                      "The same assumed throughput and power apply to Gemma, Qwen custom runner, and Qwen Biomni. "
                      "This is modeled active energy, not measured wall-clock energy. It excludes idle time, "
                      "CPU/RAM, cooling, hardware depreciation and other infrastructure; assumes no batching "
                      "or prompt-cache speedup. Cached-input discounts apply only to API pricing.", y, small)
        y = paragraph("<b>Coverage and reproduction:</b> OpenAI token buckets come from provider audit files; "
                      "Claude known usage is reconstructed from call journals. Local-model electricity uses "
                      "the run inventory's input/output totals; Biomni uses its exported usage. Unknown token "
                      "usage is not imputed. Missing calls and unlogged retries can make estimates lower bounds, "
                      "including for completed runs. Costs use the selected run inventory, independent of the "
                      "scientific-score cohort. Unchanged terminal API usage is cached by the complete run row; "
                      "use --refresh-costs after changing audit files in place. Configurable assumptions: "
                      "--cost-assumptions PATH. Outputs: cost_estimates.json, run_costs.csv, cost_totals.csv, "
                      "cost_by_workflow.csv and cost_by_condition.csv.", y, small)
        paragraph('<b>Rate sources:</b> <link href="https://developers.openai.com/api/docs/pricing" color="#2575ad">'
                  'OpenAI API pricing</link>; <link href="https://platform.claude.com/docs/en/about-claude/pricing" '
                  'color="#2575ad">Anthropic API pricing</link>. These are API-equivalent estimates, not invoice totals. '
                  'An external GCP billing check was approximately $4,400 for completed Opus runs; this is '
                  'not used to calibrate the token-based calculation.', y, small)
        c.showPage()
        c.save()

        # Merge original PDF pages as vectors, on uniform 16:9 pages with continuous numbering.
        writer = PdfWriter()
        sections = [("Overview, discoveries, methods and tables", PdfReader(narrative)),
                    ("Persistent + Biomni figures", persistent),
                    ("All-workflow figures by metric", prior),
                    ("Figures by model and harness", grouped)]
        for title, reader in sections:
            writer.add_outline_item(title, len(writer.pages))
            for page in reader.pages:
                target = writer.add_blank_page(width=width, height=height)
                scale = min(width / float(page.mediabox.width),
                            (height - 22) / float(page.mediabox.height))
                dx = (width - float(page.mediabox.width) * scale) / 2
                target.merge_transformed_page(page, Transformation().scale(scale).translate(dx, 22))

        appendix = io.BytesIO()
        c = canvas.Canvas(appendix, pagesize=(width, height))
        heading("Source files and model identifiers", "Frozen task definitions, saved result snapshot, and reproducible outputs")
        y = paragraph(f"Dataset pair: <b>{context['pair_id']}</b><br/>"
                      f"Biomni export generated: {metadata['biomni_generated_at']}<br/>"
                      "The report uses the saved result snapshot; rebuilding a PDF alone does not "
                      "refresh ongoing experiments. Source hashes are recorded in "
                      "clinical_context.json and provenance.json.", 538, small)
        for source in context["sources"]:
            y = paragraph(escape(source["path"]), y, tiny)
        y = draw_table([["<b>Display name</b>", "<b>Recorded model identifier</b>"]] +
                       [[name, escape(model)] for model, name in names.items()], [220, 846], y, font=small)
        paragraph("Normal rerun: python3 scripts/expected_surprising/build_presentation_results.py<br/>"
                  "Saved-snapshot PDF rebuild: add --report-only. Redraw figures and rebuild the PDF: "
                  "add --figures-only.<br/>"
                  "Additional methods, source selection, aliases, metric availability, and version "
                  "provenance: README.md, run_metrics.csv, condition_metrics.csv, and clinical_context.json.",
                  y, small)
        c.save()
        appendix.seek(0)
        writer.add_outline_item("Source files and model identifiers", len(writer.pages))
        writer.add_page(PdfReader(appendix).pages[0])
        total = len(writer.pages)
        for i, page in enumerate(writer.pages, 1):
            footer = io.BytesIO()
            fc = canvas.Canvas(footer, pagesize=(width, height))
            fc.setFillColor(muted)
            fc.setFont("ReportSans", 8)
            fc.drawString(margin, 10, "NSCLC clinical experiment report")
            fc.drawRightString(width - margin, 10, f"{i} / {total}")
            fc.save()
            footer.seek(0)
            page.merge_page(PdfReader(footer).pages[0])
        writer.add_metadata({"/Title": "Clinical experiment report", "/Author": "Oncology Co-Scientist"})
        destination = out / "clinical_experiment_report.pdf"
        temporary = out / ".clinical_experiment_report.tmp.pdf"
        writer.write(temporary)
        temporary.replace(destination)
        print(f"Combined report: {destination} ({total} pages)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    build(args.out.resolve(), args.repo.resolve(), json.loads(args.sources.read_text()))
