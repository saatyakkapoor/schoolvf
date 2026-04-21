/**
 * Play a short sound when a plate is detected (WebSocket `detection` message).
 * Uses `public/sounds/Detected.mp3` if present; otherwise a short synthesized beep
 * (browsers may require a click on the page once before audio works).
 */

let mp3: HTMLAudioElement | null = null;
let mp3Failed = false;

function playBeepFallback(): void {
  try {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;
    gain.gain.value = 0.12;
    osc.start();
    osc.stop(ctx.currentTime + 0.14);
  } catch {
    /* autoplay / AudioContext blocked until user gesture */
  }
}

/** Call when the API pushes a new plate detection over the live WebSocket. */
export function playPlateDetectedSound(): void {
  if (!mp3Failed) {
    if (!mp3) {
      mp3 = new Audio("/sounds/Detected.mp3");
      mp3.preload = "auto";
    }
    mp3.currentTime = 0;
    void mp3.play().catch(() => {
      mp3Failed = true;
      playBeepFallback();
    });
    return;
  }
  playBeepFallback();
}
