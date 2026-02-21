/**
 * Mirror of scouter/env/game_logic.py action encoding, in JavaScript.
 */

// MAX_HAND = 14: 11 cards dealt + up to 3 scouted (one per chip).
// Must match scouter/env/game_logic.py MAX_HAND exactly.
export const MAX_HAND = 14
export const N_SHOW = MAX_HAND * MAX_HAND   // 196
export const N_SCOUT = 2 * 2 * (MAX_HAND + 1) // 60
export const ORIENT_KEEP = N_SHOW + N_SCOUT
export const ORIENT_FLIP = N_SHOW + N_SCOUT + 1
export const MAX_ACTIONS = N_SHOW + N_SCOUT + 2   // 258

export function encodeShow(start, end) {
  return start * MAX_HAND + end
}

export function decodeShow(action) {
  return [Math.floor(action / MAX_HAND), action % MAX_HAND]
}

export function encodeScout(side, flip, insertPos) {
  return N_SHOW + side * 2 * (MAX_HAND + 1) + flip * (MAX_HAND + 1) + insertPos
}

export function decodeScout(action) {
  if (action < N_SHOW || action >= (N_SHOW + N_SCOUT)) throw new Error('Not a scout action')
  const idx = action - N_SHOW
  const side = Math.floor(idx / (2 * (MAX_HAND + 1)))
  const remainder = idx % (2 * (MAX_HAND + 1))
  const flip = Math.floor(remainder / (MAX_HAND + 1))
  const insertPos = remainder % (MAX_HAND + 1)
  return [side, flip, insertPos]
}

export function isShowAction(action) {
  return action >= 0 && action < N_SHOW
}

export function isScoutAction(action) {
  return action >= N_SHOW && action < (N_SHOW + N_SCOUT)
}

export function isOrientationAction(action) {
  return action === ORIENT_KEEP || action === ORIENT_FLIP
}

/**
 * Get all valid show slices from the mask.
 * Returns [{start, end, vals}] where vals are extracted from handCards.
 */
export function validShowsFromMask(mask, handCards) {
  const shows = []
  for (let i = 0; i < N_SHOW; i++) {
    if (!mask[i]) continue
    const [start, end] = decodeShow(i)
    if (start > end || end >= handCards.length) continue
    const vals = handCards.slice(start, end + 1).map(([a, b, f]) => f ? b : a)
    shows.push({ start, end, vals, actionIdx: i })
  }
  return shows
}

/**
 * Get all valid scout options from the mask (de-duped by side).
 * Returns [{side, flip, insertPos, actionIdx}]
 */
export function validScoutsFromMask(mask) {
  const scouts = []
  for (let i = N_SHOW; i < N_SHOW + N_SCOUT; i++) {
    if (!mask[i]) continue
    const [side, flip, insertPos] = decodeScout(i)
    scouts.push({ side, flip, insertPos, actionIdx: i })
  }
  return scouts
}
