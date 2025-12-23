#ifndef API_SERVER_H
#define API_SERVER_H

#include <stdbool.h>

// Initialize the HTTP API server
// Returns true on success, false on failure
bool api_server_init(const char* port);

// Shutdown the HTTP API server
void api_server_shutdown(void);

// Poll the server for new requests (non-blocking)
// Should be called regularly from the main loop
void api_server_poll(int timeout_ms);

// Check if the server is running
bool api_server_is_running(void);

// Notify server that a game step has completed
void api_server_on_step_complete(void);

#endif // API_SERVER_H