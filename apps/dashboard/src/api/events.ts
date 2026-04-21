import apiClient from "./client";
import type { GateEvent, EventFilters, PaginatedResponse } from "../types";

export async function getEvents(
  filters?: EventFilters,
): Promise<PaginatedResponse<GateEvent>> {
  const { data } = await apiClient.get<PaginatedResponse<GateEvent>>(
    "/events",
    { params: filters },
  );
  return data;
}

export async function getEvent(id: string): Promise<GateEvent> {
  const { data } = await apiClient.get<GateEvent>(`/events/${id}`);
  return data;
}
