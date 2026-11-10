"""
Evergreen LAI fix for the Tuolumne runs (2026-11-06, agreed with the user: "switch to monTable", "evergreen should be near constant").
Found:
  - modelDecisions had LAI_method specified -> SUMMA uses LAI = summerLAI x a HARD-CODED crop green-up curve (summa_setup.f90:194: Jan 0.01 ... Jun-Aug ~0.9),
    so the conifer forest was leafless Nov-Apr (winter VAI = winterSAI ~1-1.6).
  - The monthly MODIS table in TBL_MPTABLE.TBL (also in both 0_base_settings templates) had its Evergreen Needleleaf (type 1) LAIM row edited to
    1.6 Jan-Apr / 5.3-5.5 May-Aug / 2.2 Oct-Dec. The original (still in the file, commented) is near-flat 5.1-6.4.
Fix, applied to each run's OWN copies in the work dir (the shared settings are untouched while round 6 runs):
  LAI_method specified -> monTable; type-1 LAIM column -> the original values below. SAIM unchanged (type 1: 0.3-0.7).
Side effect: with monTable the per-HRU summerLAI / winterSAI trial params are ignored; LAI comes from the table by vegTypeIndex. Shrubland (type 7) HRUs
then get LAIM 1.0-3.0 (+SAIM 0.2-0.8) instead of their per-HRU summerLAI 2.7-3.7.
usage: from g35_lai_fix import apply; apply(work_dir)   (call after the runner has written modelDecisions.txt)
"""
import re
from pathlib import Path

ENF_LAIM = [5.1, 5.0, 5.1, 5.3, 5.9, 6.3, 6.4, 6.1, 6.0, 5.5, 5.2, 5.1]       # original MODIS Evergreen Needleleaf monthly LAI (commented row in the table)


def set_enf_laim(tbl, values=ENF_LAIM, veg=1):
    """Replace the type-`veg` column of the uncommented LAIM block in the MODIS section."""
    L = Path(tbl).read_text().splitlines(keepends=True)
    i0 = next(i for i, l in enumerate(L) if "&noah_mp_modis_parameters" in l)
    i = next(i for i in range(i0, len(L)) if re.match(r"^\s*LAIM\s*=", L[i]))
    n = 0
    while n < 12:
        line = L[i]; head, body = (line.split("=", 1) if n == 0 else ("", line))
        vals = body.rstrip("\n").split(",")
        assert len(vals) >= veg and re.match(r"^\s*-?\d+\.\d+\s*$", vals[veg - 1]), f"unexpected LAIM row: {line!r}"
        w = len(vals[veg - 1]); vals[veg - 1] = f"{values[n]:.1f}".rjust(w)
        L[i] = (head + "=" if n == 0 else "") + ",".join(vals) + "\n"; n += 1; i += 1
    Path(tbl).write_text("".join(L))


def apply(work):
    work = Path(work); md = work/"modelDecisions.txt"; txt = md.read_text()
    assert "LAI_method     specified" in txt, "unexpected LAI_method line"
    txt = "\n".join("LAI_method     monTable                  ! monthly table by vegTypeIndex (evergreen row restored, g35_lai_fix.py)"
                    if l.startswith("LAI_method") else l for l in txt.splitlines()) + "\n"
    md.write_text(txt)
    set_enf_laim(work/"TBL_MPTABLE.TBL")


if __name__ == "__main__":                     # self-test on a scratch copy
    import shutil, sys, tempfile
    src = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA")
    d = Path(tempfile.mkdtemp(dir=sys.argv[1] if len(sys.argv) > 1 else None))
    for f in ("modelDecisions.txt", "TBL_MPTABLE.TBL"): shutil.copy2(src/f, d/f)
    apply(d)
    import subprocess; print(subprocess.run(["diff", str(src/"TBL_MPTABLE.TBL"), str(d/"TBL_MPTABLE.TBL")], capture_output=True, text=True).stdout)
    print([l for l in (d/"modelDecisions.txt").read_text().splitlines() if "LAI_method" in l]); shutil.rmtree(d)
