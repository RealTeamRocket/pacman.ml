#include "api_server.h"
#include "api_game.h"
#include "mongoose.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct mg_mgr mgr;
static bool server_running = false;
static struct mg_connection *pending_step_conn = NULL;

// Helper to send JSON response with Keep-Alive
static void send_json_response(struct mg_connection *c, int status,
                               const char *json) {
  mg_http_reply(c, status,
                "Content-Type: application/json\r\n"
                "Connection: keep-alive\r\n"
                "Keep-Alive: timeout=60, max=1000\r\n",
                "%s", json);
}

// Helper to send error response
static void send_error(struct mg_connection *c, int status,
                       const char *message) {
  char json[256];
  snprintf(json, sizeof(json), "{\"error\":\"%s\"}", message);
  send_json_response(c, status, json);
}

// Helper to send success response
static void send_success(struct mg_connection *c, const char *message) {
  char json[256];
  snprintf(json, sizeof(json), "{\"success\":true,\"message\":\"%s\"}",
           message);
  send_json_response(c, 200, json);
}

// Helper to parse direction from JSON
static api_direction_t parse_direction(const char *dir_str) {
  if (strcmp(dir_str, "up") == 0)
    return API_DIR_UP;
  if (strcmp(dir_str, "down") == 0)
    return API_DIR_DOWN;
  if (strcmp(dir_str, "left") == 0)
    return API_DIR_LEFT;
  if (strcmp(dir_str, "right") == 0)
    return API_DIR_RIGHT;
  if (strcmp(dir_str, "none") == 0 || strcmp(dir_str, "noinput") == 0)
    return API_DIR_NONE;
  return API_DIR_NONE;
}

// Build state JSON response
static void build_state_json(char *buffer, size_t buffer_size,
                             const api_game_state_t *state) {
  char *ptr = buffer;
  size_t remaining = buffer_size;
  int written;

  written = snprintf(ptr, remaining, "{");
  ptr += written;
  remaining -= written;

  // Pacman
  written = snprintf(ptr, remaining,
                     "\"pacman\":{\"x\":%d,\"y\":%d,\"dir\":%d,\"alive\":%s,"
                     "\"just_died\":%s},",
                     state->pacman.x, state->pacman.y, state->pacman.dir,
                     state->pacman.alive ? "true" : "false",
                     state->pacman.just_died ? "true" : "false");
  ptr += written;
  remaining -= written;

  // Ghosts
  written = snprintf(ptr, remaining, "\"ghosts\":[");
  ptr += written;
  remaining -= written;

  for (int i = 0; i < 4; i++) {
    written = snprintf(
        ptr, remaining,
        "{\"x\":%d,\"y\":%d,\"dir\":%d,\"state\":%d,\"type\":%d}%s",
        state->ghosts[i].x, state->ghosts[i].y, state->ghosts[i].dir,
        state->ghosts[i].state, state->ghosts[i].type, (i < 3) ? "," : "");
    ptr += written;
    remaining -= written;
  }

  written = snprintf(ptr, remaining, "],");
  ptr += written;
  remaining -= written;

  // Fruit
  written = snprintf(ptr, remaining,
                     "\"fruit\":{\"active\":%s,\"x\":%d,\"y\":%d,\"type\":%d},",
                     state->fruit.active ? "true" : "false", state->fruit.x,
                     state->fruit.y, state->fruit.type);
  ptr += written;
  remaining -= written;

  // Status
  written = snprintf(
      ptr, remaining,
      "\"status\":{\"score\":%u,\"hiscore\":%u,\"lives\":%d,\"round\":%d,"
      "\"dots_remaining\":%d,\"game_over\":%s,\"round_won\":%s,\"started\":%s}"
      ",",
      state->status.score, state->status.hiscore, state->status.lives,
      state->status.round, state->status.dots_remaining,
      state->status.game_over ? "true" : "false",
      state->status.round_won ? "true" : "false",
      state->status.started ? "true" : "false");
  ptr += written;
  remaining -= written;

  // Map
  written = snprintf(ptr, remaining, "\"map\":[");
  ptr += written;
  remaining -= written;

  for (int y = 0; y < API_MAP_HEIGHT; y++) {
    written = snprintf(ptr, remaining, "[");
    ptr += written;
    remaining -= written;
    for (int x = 0; x < API_MAP_WIDTH; x++) {
      written = snprintf(ptr, remaining, "%d%s", state->map[y][x],
                         (x < API_MAP_WIDTH - 1) ? "," : "");
      ptr += written;
      remaining -= written;
    }
    written =
        snprintf(ptr, remaining, "]%s", (y < API_MAP_HEIGHT - 1) ? "," : "");
    ptr += written;
    remaining -= written;
  }

  written = snprintf(ptr, remaining, "],");
  ptr += written;
  remaining -= written;

  // Tick
  written = snprintf(ptr, remaining, "\"tick\":%u}", state->tick);
}

// Notify server that a game step has completed
void api_server_on_step_complete(void) {
  if (pending_step_conn) {
    api_game_state_t state = api_game_get_state();
    char response[4096];
    build_state_json(response, sizeof(response), &state);
    send_json_response(pending_step_conn, 200, response);
    pending_step_conn = NULL;
  }
}

// HTTP event handler
static void http_handler(struct mg_connection *c, int ev, void *ev_data) {
  if (ev == MG_EV_CLOSE) {
      if (c == pending_step_conn) {
          pending_step_conn = NULL;
      }
      return;
  }
  if (ev == MG_EV_HTTP_MSG) {
    struct mg_http_message *hm = (struct mg_http_message *)ev_data;

    // Enable Keep-Alive for this connection
    c->is_draining = 0;

    // POST /api/start - Start the game
    if (mg_match(hm->uri, mg_str("/api/start"), NULL)) {
      if (mg_strcmp(hm->method, mg_str("POST")) == 0) {
        api_game_start();
        send_success(c, "Game started");
      } else {
        send_error(c, 405, "Method not allowed");
      }
    }
    // POST /api/restart - Restart the game
    else if (mg_match(hm->uri, mg_str("/api/restart"), NULL)) {
      if (mg_strcmp(hm->method, mg_str("POST")) == 0) {
        api_game_restart();
        send_success(c, "Game restarted");
      } else {
        send_error(c, 405, "Method not allowed");
      }
    }
    // POST /api/step - Step the game with direction
    else if (mg_match(hm->uri, mg_str("/api/step"), NULL)) {
      if (mg_strcmp(hm->method, mg_str("POST")) == 0) {
        // Parse JSON body for direction
        char direction[16] = "none";
        struct mg_str json = hm->body;

        // Simple JSON parsing for direction field
        double dir_num = 0;
        if (mg_json_get_num(json, "$.direction", &dir_num)) {
          // Direction as number: 0=none, 1=up, 2=down, 3=left, 4=right
          api_direction_t dir = (api_direction_t)((int)dir_num);
          if (dir < 0 || dir > API_DIR_RIGHT)
            dir = API_DIR_NONE;
          api_game_step(dir);
        } else {
          // Try to get direction as string
          char *dir_str = mg_json_get_str(json, "$.direction");
          if (dir_str != NULL) {
            strncpy(direction, dir_str, sizeof(direction) - 1);
            direction[sizeof(direction) - 1] = '\0';
            free(dir_str);
          } else {
            // No direction provided, use none
            strcpy(direction, "none");
          }
          api_direction_t dir = parse_direction(direction);
          api_game_step(dir);
        }

        // Store connection and wait for step completion
        // The response will be sent in api_server_on_step_complete()
        if (pending_step_conn != NULL) {
            // If there's already a pending connection, error it out? 
            // Or just replace it? Let's error the old one to be safe/clean
            send_error(pending_step_conn, 503, "New step request superseded");
        }
        pending_step_conn = c;
        
        // We DO NOT send a response here.
      } else {
        send_error(c, 405, "Method not allowed");
      }
    }
    // GET /api/state - Get current game state
    else if (mg_match(hm->uri, mg_str("/api/state"), NULL)) {
      if (mg_strcmp(hm->method, mg_str("GET")) == 0) {
        api_game_state_t state = api_game_get_state();
        char response[4096];
        build_state_json(response, sizeof(response), &state);
        send_json_response(c, 200, response);
      } else {
        send_error(c, 405, "Method not allowed");
      }
    }
    // GET /health - Health check endpoint
    else if (mg_match(hm->uri, mg_str("/health"), NULL)) {
      send_success(c, "OK");
    }
    // 404 Not Found
    else {
      send_error(c, 404, "Endpoint not found");
    }
  }
}

bool api_server_init(const char *port) {
  if (server_running) {
    fprintf(stderr, "API server already running\n");
    return false;
  }

  // Disable Mongoose logging for performance
  mg_log_set(0);

  mg_mgr_init(&mgr);

  char addr[64];
  snprintf(addr, sizeof(addr), "http://0.0.0.0:%s", port);

  struct mg_connection *c = mg_http_listen(&mgr, addr, http_handler, NULL);
  if (c == NULL) {
    fprintf(stderr, "Failed to create HTTP listener on %s\n", addr);
    mg_mgr_free(&mgr);
    return false;
  }

  // Enable Keep-Alive by default for all connections
  c->is_resp = 0;

  server_running = true;
  printf("HTTP API server listening on %s\n", addr);
  printf("Endpoints:\n");
  printf("  POST /api/start    - Start the game\n");
  printf("  POST /api/restart  - Restart the game\n");
  printf("  POST /api/step     - Step the game (body: {\"direction\": "
         "\"up|down|left|right|none\"})\n");
  printf("  GET  /api/state    - Get current game state\n");
  printf("  GET  /health       - Health check\n");

  return true;
}

void api_server_shutdown(void) {
  if (server_running) {
    mg_mgr_free(&mgr);
    server_running = false;
    printf("HTTP API server stopped\n");
  }
}

void api_server_poll(int timeout_ms) {
  if (server_running) {
    mg_mgr_poll(&mgr, timeout_ms);
  }
}

bool api_server_is_running(void) { return server_running; }
