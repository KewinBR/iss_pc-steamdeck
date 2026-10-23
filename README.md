<p align="center"><img src="https://raw.githubusercontent.com/KewinBR/infamous-second-son-steamdeck/main/arte/logo-vertical.png" width="360" alt="inFAMOUS Second Son - Steamdeck Version"></p>

# inFAMOUS Second Son: Steamdeck Version (source code)

**To install and play, go to [infamous-second-son-steamdeck](https://github.com/KewinBR/infamous-second-son-steamdeck)**
(installer, tutorial in English and Portuguese, credits).

This repository is the source code of the port that runs there: a copy of HayatoG's
[iss_pc](https://github.com/HayatoG/iss_pc) (native PC port of inFAMOUS Second Son, with renderer v2) with the
changes made for the Steam Deck, in the `steam-deck` branch. GPL-2.0, like iss_pc. The original project's README is
in [README-iss_pc.md](README-iss_pc.md).

**Para instalar e jogar, vá em [infamous-second-son-steamdeck](https://github.com/KewinBR/infamous-second-son-steamdeck)**
(instalador, tutorial em português e inglês, créditos). Este repositório é o código do port: uma cópia do
[iss_pc](https://github.com/HayatoG/iss_pc) de HayatoG com as mudanças feitas para o Steam Deck, no ramo `steam-deck`.

## Changes for the Steam Deck / Mudanças para o Steam Deck

- Unified GPU heap (B+C) and a texture-only collector: about 3.5 GB of RAM instead of 6, +9 FPS in the neon race.
- Startup screen with shader preparation, save dialogs in the game's style, Deck A/B glyphs.
- Skip videos (experimental, Extras tab of the port menu, off by default), 16:10 video crop.
- The port menu keeps the launcher's `tlg.ini` keys (renderer, native passes) when it saves.
- Diagnostics used during the work (hunt notebook, label watch, TSC scan), all off unless their variable is set.

Steam Deck port: Kewin Lima. Base: iss_pc by HayatoG and the projects it builds on (see README-iss_pc.md).
