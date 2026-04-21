import type { LiveDebugEntry, LiveDetection } from "../types";

import apiClient from "./client";

export async function getLiveRecent(limit = 100): Promise<LiveDetection[]> {
  const { data } = await apiClient.get<LiveDetection[]>("/live/recent", {
    params: { limit },
  });
  return data;
}

export async function getLiveDebug(limit = 150): Promise<LiveDebugEntry[]> {
  const { data } = await apiClient.get<LiveDebugEntry[]>("/live/debug", {
    params: { limit },
  });
  return data;
}

export async function adjustDetection(
  eventId: string,
  payload: { swap_type: string; notes?: string | null },
): Promise<void> {
  await apiClient.post(`/live/detections/${eventId}/adjust`, {
    swap_type: payload.swap_type,
    notes: payload.notes ?? null,
  });
}
