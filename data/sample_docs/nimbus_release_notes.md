# Nimbus N7 — Firmware Release Notes ("Cirrus")

Firmware for the N7 is released under the codename "Cirrus" on the first Tuesday
of every month. Updates are installed from the Halo portal → Devices → Firmware.
A drone must be docked and above 40% charge before an update can start.

## Cirrus 5.3 (September 2024)
- New "quiet hours" mode limits rotor speed between 22:00 and 06:00, reducing
  noise in warehouses that run night shifts.
- Inventory scans now upload in batches of 250 records instead of 100, cutting
  sync time for large sites by roughly a third.
- Fixed a bug where the dock LED stayed amber after a completed charge.

## Cirrus 5.2 (August 2024)
- Added support for a second geofence file, `zone_overrides.cfg`, so sites can
  temporarily close an aisle without editing the main `zone.cfg`.
- Return-to-dock now triggers automatically when charge falls to 20%.
- Improved RFID read rate on metal shelving from 91% to 97% in internal tests.

## Cirrus 5.1 (July 2024)
- Introduced fleet scheduling: up to 12 drones can share one scan plan.
- The Halo portal now shows a per-drone charge-cycle counter, making it easier
  to spot packs approaching the 500-cycle replacement point.

## Cirrus 5.0 (June 2024)
- Major release. Replaced the obstacle-avoidance model; the drone now holds a
  minimum clearance of 0.8 m from detected people.
- Dropped support for docks built before 2021 (hardware revision A).
- Scanner calibration data is now stored on the drone, so recalibration
  survives a factory reset.

## Rollback policy
If an update causes problems, a site can roll back to the previous Cirrus
version within 14 days from the Halo portal. Rollbacks older than one version
require a support ticket.
