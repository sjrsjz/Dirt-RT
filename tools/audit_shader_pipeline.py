"""Compile every entry and validate SPIR-V; artifacts stay under temp/.

The relaxed Vulkan flags emulate the host's binding/location assignment, not
its runtime descriptor layout. This is a static check, not a frame benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess

from shader_compile import GLSLANG, ROOT, expand


STAGES = {'.csh': 'comp', '.fsh': 'frag', '.vsh': 'vert',
          '.rgen': 'rgen', '.rchit': 'rchit', '.rahit': 'rahit', '.rmiss': 'rmiss'}


def compile_entry(path, folder, defines):
    stage = STAGES[path.suffix]
    source = expand(path)
    # Iris options replace their source #define; a command-line -D would be
    # overwritten (or rejected) by settings.glsl and would not test the option.
    injected = []
    for define in defines:
        name, _, value = define.partition('=')
        pattern = r'^\s*#define[ \t]+' + re.escape(name) + r'(?:[ \t]+[^\n]*)?$'
        source, count = re.subn(pattern, '#define ' + name + ' ' + (value or '1'),
                                source, flags=re.M)
        if not count:
            injected.append(define)
    lines = source.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('#version'))
    version = lines.pop(index)
    target = folder / (path.name + '.glsl')
    target.write_text(version + '\n' + '\n'.join(lines) + '\n', encoding='utf-8')
    binary = folder / (path.name + '.spv')
    command = [str(GLSLANG), '-V', '-R', '-S', stage,
               '--auto-map-bindings', '--auto-map-locations',
               '-DMC_GL_AMD_gpu_shader_half_float=1',
               '--target-env', 'vulkan1.2']
    command += ['-D' + define for define in injected]
    proc = subprocess.run(command + ['-o', str(binary), str(target)],
                          capture_output=True, text=True)
    result = {'entry': path.name, 'compiled': proc.returncode == 0}
    if proc.returncode == 0:
        validation = subprocess.run([str(GLSLANG.with_name('spirv-val.exe')),
                                     '--scalar-block-layout', str(binary)],
                                    capture_output=True, text=True)
        result.update(validated=validation.returncode == 0,
                      spirv_bytes=binary.stat().st_size)
        log = proc.stdout + proc.stderr + validation.stdout + validation.stderr
    else:
        log = proc.stdout + proc.stderr
    (folder / (path.name + '.log')).write_text(log, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='current')
    parser.add_argument('-D', '--define', action='append', default=[])
    parser.add_argument('--entry', action='append', help='Entry filename glob; repeatable')
    args = parser.parse_args()
    # A GLSL reserved word cannot be used for variables, members or functions.
    for path in (ROOT / 'shaders').rglob('*'):
        if path.suffix not in set(STAGES) | {'.glsl'}:
            continue
        code = re.sub(r'/\*.*?\*/|//[^\n]*', '',
                      path.read_text(encoding='utf-8-sig'), flags=re.S)
        if re.search(r'\bpacked\b', code):
            parser.error(f'reserved GLSL identifier in {path.relative_to(ROOT)}')
    folder = (ROOT / 'temp' / 'pipeline_audit' / args.label).resolve()
    if not folder.is_relative_to((ROOT / 'temp').resolve()):
        parser.error('label must stay under temp/')
    folder.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in (ROOT / 'shaders').iterdir()
                   if p.suffix in STAGES and
                   (not args.entry or any(p.match(g) for g in args.entry)))
    if not paths:
        parser.error('no matching entries')
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda p: compile_entry(p, folder, args.define), paths))
    report = {'defines': args.define, 'entries': results}
    (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    failed = [r['entry'] for r in results if not r.get('validated')]
    print(json.dumps({'entries': len(results), 'failures': failed,
                      'report': str(folder / 'report.json')}, indent=2))
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
