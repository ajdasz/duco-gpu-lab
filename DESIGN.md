# DUCO GPU Lab visual contract

## Purpose

Audience: a technically curious miner monitoring a live WebGPU session.
Product: local mining dashboard. Personality: precise, calm, capable.
Density: medium at a glance, with SHA-1 details available on demand.
Primary action: start or stop mining. Desired feeling: clear control and trust in the reported data.

## Direction

Dominant style: Swiss / International. Use a disciplined grid, left aligned type, hairline dividers and one restrained action accent. Minimalism was the runner-up, but hiding the live measurements would make this tool less useful. The layout pattern is an instrument dashboard: status and controls first, measured output second, explanation and history below.

## Existing design system and adaptation

The previous page used a dark background, orange action color, green accepted state, red rejected state, system sans type, and CSS custom properties in `style.css`. Preserve the dark identity and semantic colors. Remove the decorative orbit, ambient gradients, glow, oversized hero, and repeated card treatments because they compete with the measurements. No third font family or external font request is needed.

Token mapping: the existing `--bg`, `--panel`, `--line`, `--text`, `--muted`, `--orange`, `--green`, and `--red` remain the source of color roles. Add spacing, radius and type tokens to replace scattered values. The orange hue is reserved for the primary action and a single chart accent; green and red retain their status meanings.

## Layout and content rules

- 1280px maximum content width, 8px spacing module, thin rules and compact 4px corners.
- One dominant hashrate figure. Hash count, accepted and rejected shares form a quiet supporting row.
- Keep the experimental use-at-your-own-risk notice visible below the header on every visit.
- Keep job progress visible. Put the sample SHA-1 calculation behind a disclosure so it remains accessible without crowding the first screen.
- Keep pool settings and the event log available. Explain technical values in plain English.
- Compose for 1440px, 768px, and 390px. The mobile view moves controls beneath the title and gives the hashrate its own row.
- Body text is at least 15px. Labels are at least 12px. Interactive targets are at least 44px high and have visible focus states.
- No decorative animation while mining. Honor reduced motion.
