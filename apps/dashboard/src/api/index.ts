export { default as apiClient } from "./client";

export {
  getCameras,
  getCamera,
  createCamera,
  updateCamera,
  probeCamera,
} from "./cameras";
export { getTrips, getTrip, getCurrentBusStatus } from "./trips";
export type { CurrentBusStatusItem } from "./trips";
export { getEvents, getEvent } from "./events";
export { getAlerts, resolveAlert } from "./alerts";
export { getPlateDetail } from "./plates";
export { createCorrection, getCorrections } from "./corrections";
export { getDashboardSummary } from "./dashboard";
export { getLiveDebug, getLiveRecent } from "./live";
export {
  getVehicles,
  getVehicle,
  createVehicle,
  updateVehicle,
  deleteVehicle,
  getVehicleByPlate,
  getRoutes,
} from "./vehicles";
export {
  getUsers,
  createUser,
  updateUser,
  deleteUser,
  getMyProfile,
} from "./users";
