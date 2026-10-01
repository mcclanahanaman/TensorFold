"""The kernel classes of ``prefill_profile.py``'s labels (no torch: the summary reads them anywhere)."""

from __future__ import annotations

CLASSES = ("routed GEMM (FI W4A16)", "routed GEMM (portable, MTP layer)", "routed glue (route/gather/scatter)",
           "shared expert", "dense prompt GEMMs", "QSA / attention", "GDN prefill", "norms / elementwise",
           "host-side steps (stage, sample)", "host gaps / idle")


def klass(path: tuple) -> str:
    """The kernel class of a label path (innermost label last)."""

    if not path:
        return "host gaps / idle"
    leaf = path[-1]
    if leaf.startswith("dense."):
        return "dense prompt GEMMs"
    if leaf in ("moe.routed.gemm1", "moe.routed.gemm2"):
        return "routed GEMM (FI W4A16)"
    if leaf.startswith("moe.portable"):
        return "routed GEMM (portable, MTP layer)"
    if leaf in ("moe.routed", "moe.router", "moe.select"):
        return "routed glue (route/gather/scatter)"
    if leaf.startswith("moe"):
        return "shared expert"                       # the shared expert's GEMMs, SwiGLU and its slot copies
    if leaf.startswith("attn"):
        return "QSA / attention"
    if leaf.startswith("gdn") or (leaf == "shift" and "gdn" in path):
        return "GDN prefill"
    if leaf in ("pass", "compute", "stage", "admit", "joined", "make_room", "mtp.stage", "absorb", "mtp"):
        return "host-side steps (stage, sample)"     # containers: their own time is host work or the GPU waiting
    return "norms / elementwise"
