#include "api_game.h"
#include <string.h>

// Forward declarations from pacman.c - these need to be exposed
// We'll need to modify pacman.c to expose these functions and the state struct

// API state
static struct {
    bool enabled;
    api_direction_t current_direction;
    bool step_requested;
} api_state = {
    .enabled = false,
    .current_direction = API_DIR_NONE,
    .step_requested = false
};

void api_game_init(void) {
    memset(&api_state, 0, sizeof(api_state));
    api_state.enabled = false;
}

void api_game_set_enabled(bool enabled) {
    api_state.enabled = enabled;
}

bool api_game_is_enabled(void) {
    return api_state.enabled;
}

void api_game_apply_input(api_direction_t direction) {
    api_state.current_direction = direction;
}

api_direction_t api_game_get_current_direction(void) {
    return api_state.current_direction;
}

void api_game_clear_direction(void) {
    api_state.current_direction = API_DIR_NONE;
}

bool api_game_should_step(void) {
    return api_state.step_requested;
}

void api_game_clear_step_request(void) {
    api_state.step_requested = false;
}

void api_game_request_step(void) {
    api_state.step_requested = true;
}

// NOTE: The following functions (api_game_start, api_game_restart, 
// api_game_step, api_game_get_state) are implemented in pacman.c
// where they have direct access to the game state.