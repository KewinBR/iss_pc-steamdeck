# SPDX-License-Identifier: GPL-2.0-or-later
"""The currently recommended setup (presets_current.py, RECOMMENDED 2: this port's renderer) on
renderer v2 (port/gpu_v2), with and without its native passes. All three list the renderer, so
applying one undoes the others' choice."""
from presets_current import CURRENT

V2_ON = {
    'id': 'renderer_v2',
    'exported': '2026-10-20',
    'names': {'en': 'RENDERER V2', 'pt': 'RENDERER V2'},
    'what': {
        'en': 'The recommended setup on renderer v2 (experimental): its own shader cache, compiled again '
              'on the first start. Native passes stay off (NVIDIA driver crash).',
        'pt': 'A configuração recomendada no renderer v2 (experimental): cache de shaders próprio, '
              'compilado de novo na primeira partida. Os native passes ficam desligados (crash no driver NVIDIA).',
    },
    'settings': dict(CURRENT['settings']),
    'items': dict(CURRENT['items'], TLG_RENDERER='2', TLG_NATIVE_PASSES='0'),
}

V2_NATIVE = {
    'id': 'renderer_v2_native',
    'exported': '2026-10-20',
    'names': {'en': 'RENDERER V2 + NATIVE PASSES', 'pt': 'RENDERER V2 + NATIVE PASSES'},
    'what': {
        'en': 'RENDERER V2 with its native passes on (for tests): more frames in some places, but on NVIDIA\'s '
              'Linux driver frame drops and crashes.',
        'pt': 'RENDERER V2 com os native passes ligados (para testes): mais quadros em alguns lugares, mas no '
              'driver NVIDIA do Linux quedas de quadros e travamentos.',
    },
    'settings': dict(CURRENT['settings']),
    'items': dict(CURRENT['items'], TLG_RENDERER='2', TLG_NATIVE_PASSES='1'),
}

V2_PRESETS = (V2_ON, V2_NATIVE)
