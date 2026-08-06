#  This source file is part of the Avogadro project
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-auto3d/

"""Dispatch feature identifier to the appropriate handler."""

import json
import os
import sys


def run(args):
    avo_input = json.load(sys.stdin)

    # Preserve a clean channel for the JSON result, then redirect the real
    # stdout file descriptor to stderr. Auto3D, PyTorch and its multiprocessing
    # workers print banners, progress bars and model-download chatter (some from
    # C/Fortran and some at teardown) that would otherwise corrupt the single
    # JSON object Avogadro expects on stdout. This is never restored: the process
    # exits right after we emit.
    sys.stdout.flush()
    result_fd = os.dup(1)
    os.dup2(2, 1)

    output = None
    match args.feature:
        case "conformers":
            from .conformers import generate
            output = generate(avo_input)

    if output is not None:
        data = (json.dumps(output) + "\n").encode()
        while data:
            data = data[os.write(result_fd, data):]
