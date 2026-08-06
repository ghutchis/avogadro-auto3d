#  This source file is part of the Avogadro project
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-auto3d/

"""Conformer generation with Auto3D's neural-network-potential pipeline.

Receives an Avogadro molecule (3D, explicit hydrogens) as an SDF molblock, runs
Auto3D on it, and returns the ranked low-energy conformers as a multi-conformer
SDF (each tagged with its relative energy). Auto3D is told NOT to enumerate
stereoisomers -- Avogadro already supplies a defined 3D configuration, so the
input stereochemistry is preserved. Tautomer enumeration is offered as an option.
"""

import os
import tempfile
from io import StringIO

from rdkit import Chem

# Auto3D reports total energies (E_tot) in Hartree; Avogadro plots conformer
# energies in kJ/mol.
_HARTREE_TO_KJ = 2625.4996


def _mol_energy(mol) -> float:
    """Total energy (Hartree) tagged by Auto3D, or +inf if unavailable."""
    try:
        return float(mol.GetProp("E_tot"))
    except (KeyError, ValueError):
        return float("inf")


def generate(avo_input: dict) -> dict:
    """Run Auto3D on the current molecule and return conformers as SDF."""
    options = avo_input.get("options", {})
    method = options.get("method", "AIMNET")
    k = max(1, int(options.get("conformers", 10)))
    enumerate_tautomer = bool(options.get("tautomers", False))

    # Parse the incoming structure, keeping explicit hydrogens and the 3D
    # geometry that carries the stereochemistry.
    mol = Chem.MolFromMolBlock(avo_input.get("sdf", ""), removeHs=False)
    if mol is None:
        return {"error": "Could not interpret the molecule."}

    # Lock in the configuration implied by the supplied coordinates so Auto3D
    # (with enumerate_isomer=False) preserves it rather than guessing.
    Chem.AssignStereochemistryFrom3D(mol)
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    if not name.strip():
        mol.SetProp("_Name", "avogadro")

    # Enable the GPU only when CUDA is actually available through torch.
    try:
        import torch
        use_gpu = bool(torch.cuda.is_available())
    except Exception:
        use_gpu = False

    # Auto3D imports are deferred so a plain dispatch stays light.
    from Auto3D import Auto3DOptions, main

    with tempfile.TemporaryDirectory() as tmp:
        input_sdf = os.path.join(tmp, "input.sdf")
        writer = Chem.SDWriter(input_sdf)
        writer.write(mol)
        writer.close()

        config = Auto3DOptions(
            path=input_sdf,
            k=k,
            enumerate_isomer=False,             # keep the user's stereochemistry
            enumerate_tautomer=enumerate_tautomer,
            optimizing_engine=method,
            use_gpu=use_gpu,
            gpu_idx=0,
            job_name="avogadro",
        )

        try:
            out_path = str(main(config))
        except SystemExit as exc:
            # main() exits if input validation fails or nothing converges.
            return {"error": f"Auto3D produced no structures ({exc})."}
        except Exception as exc:
            return {"error": f"Auto3D failed:\n{exc}"}

        out_mols = [m for m in Chem.SDMolSupplier(out_path, removeHs=False)
                    if m is not None]

    if not out_mols:
        return {"error": "Auto3D returned no conformers."}

    # Rank by total energy and tag each conformer with its energy relative to
    # the lowest (kJ/mol) so Avogadro can plot the conformer energies. Tautomers
    # share the input's molecular formula, so every block has the same atom
    # count and Avogadro reads them as a single conformer set.
    out_mols.sort(key=_mol_energy)
    e_min = _mol_energy(out_mols[0])

    sio = StringIO()
    writer = Chem.SDWriter(sio)
    for m in out_mols:
        energy = _mol_energy(m)
        if energy != float("inf") and e_min != float("inf"):
            m.SetDoubleProp("energy", (energy - e_min) * _HARTREE_TO_KJ)
        elif m.HasProp("energy"):
            m.ClearProp("energy")
        writer.write(m)
    writer.close()

    return {"moleculeFormat": "sdf", "sdf": sio.getvalue()}
