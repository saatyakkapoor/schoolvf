/**
 * Live WebSocket state is owned by {@link LiveWebSocketProvider} on MainLayout
 * so Overview, Live feed, and debug panel share one connection.
 */
export { LiveWebSocketProvider, useLiveWebSocket } from "../context/LiveWebSocketContext";
export type { LiveWebSocketValue } from "../context/LiveWebSocketContext";
