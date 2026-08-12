# The Mughal-city prompt, all versions

Kept here (not in scratchpad or on a pod) so it survives — the pod-side copies died
with the terminated boxes.

---

## v0 — yours, original

> A photorealistic bustling mughal city with old Dravida architecture built as tall as
> skyscrappers at dusk in rural india by the lake

Used on **Marble 1.1** as-is (Marble takes text directly).

---

## v1 — expanded, static description

Used for the six `MatrixGame3.5(p)/clips/city_case_*.mp4` and to generate the anchor image.

> A photorealistic bustling Mughal city with old Dravida architecture built as tall as
> skyscrapers, at dusk, in rural India, by the lake. Towering carved gopuram-like stone
> towers rise in tiered layers above the water, their sandstone and granite surfaces
> catching the last warm orange light of the setting sun. Domes, arched iwans and latticed
> jharokha balconies stack upward into hazy blue evening sky. Crowds move along the ghats
> at the lake edge, small boats drift on still reflective water, lanterns and cooking fires
> glow amber between the buildings. Smoke and dust hang in the humid air, birds circle the
> high towers.

---

## v2 — motion-heavy

Used for `MatrixGame3.5(p)/tour/palace_tour_6b.mp4`, `Wan2.2(t)/`, and the ABot batch run.
Written after the first clips came back too static.

> A photorealistic bustling Mughal city with old Dravida architecture built as tall as
> skyscrapers, at dusk, in rural India, by the lake. Towering carved gopuram stone towers
> rise in tiered layers above the water, sandstone and granite catching the last warm
> orange light. The camera glides steadily forward along the waterfront, sweeping past
> carved facades, arched iwans and latticed jharokha balconies. Wooden boats rock and drift
> on the water, oars dipping, wakes rippling outward across the reflective surface.
> Hundreds of oil lamps and lanterns flicker and pulse along the ghats, reflections
> shimmering and breaking on the moving water. Crowds surge along the stone steps, figures
> walking and gesturing. Banks of mist and smoke roll through the streets and around the
> towers, pushed by gusting wind, curling and dissipating. Flocks of birds wheel across the
> darkening sky. Cloth awnings and banners snap and billow in the wind. Everything is in
> continuous motion.

---

## v3 — for ABot's live UI  ← paste this one

ABot is **first-person and walkable**, so it wants different cues than a camera-path model.
Its own default prompt is:

> "A realistic outdoor world scene with a navigable path, natural lighting, detailed ground
> texture, and stable forward motion."

Note what that emphasises: *navigable path*, *ground texture*, *stable forward motion* —
because you're walking, not flying. v2's "the camera glides steadily forward" actively
fights that; you control the camera here. So v3 drops the camera direction, keeps the
scene dynamics, and adds walkable ground.

> A photorealistic bustling Mughal city with old Dravida architecture built as tall as
> skyscrapers, at dusk, in rural India, beside a wide lake. A broad stone walkway runs
> ahead along the water's edge, worn granite flagstones underfoot, wet in places and
> catching the light. Towering carved gopuram towers rise in tiered layers on either side,
> sandstone and granite glowing in the last warm orange sun, with arched iwans, deep
> shadowed colonnades and latticed jharokha balconies overhead. Oil lamps and lanterns
> flicker along the ghats, their reflections breaking on the moving water. Wooden boats
> rock at their moorings. Crowds move along the stone steps ahead. Warm haze and drifting
> smoke hang between the buildings, birds circle the high towers. Detailed ground texture,
> natural evening lighting, stable navigable path ahead.
