import apiClient from "./client";

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface UserMe {
  username: string;
}

export async function loginRequest(username: string, password: string): Promise<LoginResponse> {
  const { data } = await apiClient.post<LoginResponse>("/auth/login", {
    username,
    password,
  });
  return data;
}

export async function getMe(): Promise<UserMe> {
  const { data } = await apiClient.get<UserMe>("/auth/me");
  return data;
}
