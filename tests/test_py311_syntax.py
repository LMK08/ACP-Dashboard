"""CI runs Python 3.11; the local interpreter is newer. Two things 3.12+
accepts that 3.11 rejects have bitten the deploy gate: an f-string whose
own quote character reappears inside a replacement field (PEP 701), and
backslashes inside replacement fields. Scan every tracked .py file."""
import os
import re
import subprocess

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# f-string ... { ... <same quote> ... }
_NESTED_QUOTE = re.compile(r"""(?<![A-Za-z0-9_])f(['"])(?:(?!\1).)*\{[^}\n]*\1[^}\n]*\}""")


def _tracked_py():
    out = subprocess.run(['git', 'ls-files', '*.py'], cwd=DASH, capture_output=True, text=True)
    return [os.path.join(DASH, p) for p in out.stdout.split() if p.endswith('.py')]


def test_no_python312_only_fstrings():
    files = _tracked_py()
    assert files, 'git ls-files returned nothing'
    bad = []
    for path in files:
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if _NESTED_QUOTE.search(line):
                    bad.append(f"{os.path.relpath(path, DASH)}:{i}: {line.strip()[:100]}")
    assert not bad, 'nested same-quote f-strings are a SyntaxError on CI (Python 3.11):\n' + '\n'.join(bad)
