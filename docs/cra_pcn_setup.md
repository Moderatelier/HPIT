# CRA-PCN dependency setup

The repository does not redistribute CRA-PCN source code. The HPIT model loads a user-supplied checkout at runtime.

## Fixed upstream version

- Repository: <https://github.com/EasyRy/CRA-PCN>
- Commit: `e87d11a5134332366c507f7cf784a03ad13a4d00`
- Compatibility patch: `patches/cra-pcn-hpit-compatibility.patch`

The patch redirects point-cloud operations through `hpitd.point_ops`, adds the dimensions and grouped-neighbour behaviour used by HPIT, and extends the CRT return value with the reduced positions required by the reconstruction stage. It does not contain the upstream source file.

## Installation

Run these commands from the HPIT repository root:

```bash
git clone https://github.com/EasyRy/CRA-PCN.git third_party/CRA-PCN
git -C third_party/CRA-PCN checkout e87d11a5134332366c507f7cf784a03ad13a4d00
git -C third_party/CRA-PCN apply ../../patches/cra-pcn-hpit-compatibility.patch
python -m pip install -e .
python CODE/Computation/Main.py self-test --device cpu
```

When the checkout is stored elsewhere, set its root before running HPIT:

PowerShell:

```powershell
$env:HPITD_CRA_PCN_ROOT = "C:\software\CRA-PCN"
python CODE\Computation\Main.py self-test --device cpu
```

Bash:

```bash
export HPITD_CRA_PCN_ROOT=/opt/CRA-PCN
python CODE/Computation/Main.py self-test --device cpu
```

The loader verifies the compatibility marker and the extended `CRT.forward` signature. An unpatched or unexpected checkout is rejected with an actionable error.

## Update policy

Do not update the CRA-PCN commit silently. A different commit requires a fresh patch-application test, original-checkpoint compatibility check, and forward-output comparison.
