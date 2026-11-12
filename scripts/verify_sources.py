#!/usr/bin/env python3
"""Repository integrity and accidental embedded-data guard (not a complete privacy scan)."""
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parents[1]
python=root/'python'/'Python_FaceAssymmetry.py'
matlab=root/'matlab'/'MATLAB_FaceAssymmetry.m'
assert python.is_file() and matlab.is_file()
py=python.read_bytes()
m=matlab.read_bytes()
assert b"VERSION='62.0'" in py
assert b'function Results = MATLAB_FaceAssymmetry(varargin)' in m
assert b'function S=readScanCSV(filename,IDs)' in m
assert b'FaceASI:NoEmbeddedData' in m
assert len(m)<500_000, 'MATLAB unusually large: check for embedded capture payload'
assert b'TUFUTEFCIDUuMCBNQVQt' not in m, 'Embedded MAT data signature found'
for name,data in [('Python',py),('MATLAB public',m)]:
    print(f'{name}: {len(data):,} bytes; SHA256 {hashlib.sha256(data).hexdigest()}')
print('PASS public-source integrity and basic embedded-payload guard')
