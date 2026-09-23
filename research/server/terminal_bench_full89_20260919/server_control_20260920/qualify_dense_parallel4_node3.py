"""Qualify an additional server node without changing the frozen initial checks."""
import json
import pathlib
import sys

H=pathlib.Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel4_20260921')
sys.path.insert(0,str(H))
import qualify_dense_parallel4 as qualification

qualification.OUT=H/'qualification_node3'
original_save=qualification.save


def save(path,value):
    if path.parent==H:
        path=H/(path.stem+'_node3'+path.suffix)
    original_save(path,value)


qualification.save=save
qualification.main()
