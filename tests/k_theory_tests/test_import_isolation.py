import os
import subprocess
import sys


def test_k_theory_imports_in_fresh_python_process():
    """Verify that k theory imports in fresh python process."""
    environment = os.environ.copy()
    command = [sys.executable, "-c", "import motives.k_theory as kt; assert kt.VectorBundle; assert kt.Dual; assert kt.chern_character"]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_all_k_theory_modules_import_in_fresh_process():
    """Verify that all k theory modules import in fresh process."""
    modules = [
        "motives.k_theory", "motives.k_theory.chern", "motives.k_theory.chern.chern_character",
        "motives.k_theory.chern.chern_class", "motives.k_theory.objects", "motives.k_theory.objects.scheme",
        "motives.k_theory.objects.vector_bundle", "motives.k_theory.objects.dual_bundle",
        "motives.k_theory.objects.determinant_bundle", "motives.k_theory.objects.power_bundles",
        "motives.k_theory.operations", "motives.k_theory.operations.bundle_operations",
        "motives.k_theory.operations.power_operations", "motives.k_theory.operations.exact_sequences"
    ]
    code = "; ".join(f"import {module}" for module in modules)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
