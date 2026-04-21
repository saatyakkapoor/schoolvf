import apiClient from "./client";
import type { Vehicle, CreateVehiclePayload, UpdateVehiclePayload, RouteInfo } from "../types";

export async function getVehicles(activeOnly?: boolean): Promise<Vehicle[]> {
  const params: Record<string, string> = {};
  if (activeOnly) params.active_only = "true";
  const { data } = await apiClient.get<Vehicle[]>("/vehicles", { params });
  return data;
}

export async function getVehicle(id: string): Promise<Vehicle> {
  const { data } = await apiClient.get<Vehicle>(`/vehicles/${id}`);
  return data;
}

export async function createVehicle(payload: CreateVehiclePayload): Promise<Vehicle> {
  const { data } = await apiClient.post<Vehicle>("/vehicles", payload);
  return data;
}

export async function updateVehicle(id: string, payload: UpdateVehiclePayload): Promise<Vehicle> {
  const { data } = await apiClient.patch<Vehicle>(`/vehicles/${id}`, payload);
  return data;
}

export async function deleteVehicle(id: string): Promise<void> {
  await apiClient.delete(`/vehicles/${id}`);
}

export async function getVehicleByPlate(plate: string): Promise<Vehicle | null> {
  try {
    const { data } = await apiClient.get<Vehicle>(`/vehicles/by-plate/${encodeURIComponent(plate)}`);
    return data;
  } catch {
    return null;
  }
}

export async function getRoutes(): Promise<RouteInfo[]> {
  const { data } = await apiClient.get<RouteInfo[]>("/vehicles/routes");
  return data;
}
