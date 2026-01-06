# NAT Sector Updater (Aurora / IVAO) – OCC Oceanic (Shanwick/Gander)

A small Python terminal tool that fetches live NAT/OTS tracks and automatically generates an **IVAO Aurora** compatible **NAT_TRACKS.awh** file (**[HIGH AIRWAY]**), then patches your **.isc** to include it.

---

## Features

- Fetches live **NAT/OTS** from the FAA NAT page: https://notams.aim.faa.gov/nat.html
- Nice terminal UI if `rich` is installed (plain fallback otherwise)
- Displays tracks in a table/list in the terminal
- Auto-generates `NAT_TRACKS.awh`
- Automatically adds the `F;...` reference into your `.isc` under `[HIGH AIRWAY]`
- First run setup: language + `.isc` path + OCC/Oceanic include folder name
- If you provide a **folder** instead of a `.isc`, the script finds `.isc` files and lets you choose
- Option to **show/change the config file path** (menu + CLI argument)

---

## Requirements

- Python **3.10+** recommended
- Windows / Linux / macOS

Optional (recommended for better terminal UI):
```bash
pip install rich
