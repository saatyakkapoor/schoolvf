import apiClient from "./client";
import type {
  Trip,
  TripFilters,
  PaginatedResponse,
  BusStatus,
} from "../types";

export async function getTrips(
  filters?: TripFilters,
): Promise<PaginatedResponse<Trip>> {
  const { data } = await apiClient.get<PaginatedResponse<Trip>>("/trips", {
    params: filters,
  });
  return data;
}

export async function getTrip(id: string): Promise<Trip> {
  const { data } = await apiClient.get<Trip>(`/trips/${id}`);
  return data;
}

export interface CurrentBusStatusItem {
  plate_number: string;
  status: BusStatus;
  last_event_time: string | null;
  last_camera: string | null;
  current_trip_id: string | null;
  duration_outside_seconds: number | null;
}

export async function getCurrentBusStatus(): Promise<CurrentBusStatusItem[]> {
  const { data } = await apiClient.get<CurrentBusStatusItem[]>(
    "/trips/bus-status",
  );
  return data;
}
