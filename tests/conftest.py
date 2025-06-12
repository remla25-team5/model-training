import os
from collections import defaultdict


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Test the calculation of the ML score based on the test files per section.
    """

    section_prefixes = {
        'features_data': 'Features',
        'monitoring': 'Monitoring',
        'ml_infrastructure': 'ML Infrastructure',
        'model_development': 'Model Development',
        'train': 'Training'
    }

    results = defaultdict(lambda: {"total": 0, "passed": 0})

    for report in terminalreporter.getreports("passed") + terminalreporter.getreports("failed") \
            + terminalreporter.getreports("skipped"):
        path = report.nodeid.split("::")[0]
        filename = os.path.basename(path)
        prefix = filename.replace("test_", "").split(".")[0]
        section = section_prefixes.get(prefix, "Other")

        results[section]["total"] += 1
        if report.passed:
            results[section]["passed"] += 1

    # Compute and print section-wise scores
    terminalreporter.write_sep("-", "ML Test Score Report")
    section_scores = {}

    for section, counts in results.items():
        if counts["total"] == 0:
            score = 0.0
        else:
            score = counts["passed"]
        section_scores[section] = score
        terminalreporter.write_line(f"{section} Tests: {counts['passed']}/{counts['total']} → Score: {score:}")

    final_score = min(section_scores.values()) if section_scores else 0
    terminalreporter.write_line(f"\nFinal ML Test Score: {final_score}")
