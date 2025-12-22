#ifndef API_GAME_H
#define API_GAME_H

#include <stdbool.h>
#include <stdint.h>

// Direction values for API control
typedef enum {
    API_DIR_NONE = 0,
    API_DIR_UP,
    API_DIR_DOWN,
    API_DIR_LEFT,
    API_DIR_RIGHT
} api_direction_t;

// Game state information exposed via API
typedef struct {
    // Pacman info
    struct {
        int x;
        int y;
        int dir;
        bool alive;
        bool just_died;  // true if Pacman died in this tick
    } pacman;
    
    // Ghost info (4 ghosts)
    struct {
        int x;
        int y;
        int dir;
        int state; // 0=scatter, 1=chase, 2=frightened, 3=eaten
        int type;  // 0=blinky, 1=pinky, 2=inky, 3=clyde
    } ghosts[4];
    
    // Fruit info
    struct {
        bool active;
        int x;
        int y;
        int type;
    } fruit;
    
    // Game status
    struct {
        uint32_t score;
        uint32_t hiscore;
        int lives;
        int round;
        int dots_remaining;
        bool game_over;
        bool round_won;
        bool started;
    } status;
    
    // Timing
    uint32_t tick;
} api_game_state_t;

// Initialize API mode
void api_game_init(void);

// Start the game (equivalent to pressing any key at intro)
void api_game_start(void);

// Restart the game
void api_game_restart(void);

// Step the game one tick with given direction
void api_game_step(api_direction_t direction);

// Get current game state
api_game_state_t api_game_get_state(void);

// Check if API mode is enabled
bool api_game_is_enabled(void);

// Set API mode enabled/disabled
void api_game_set_enabled(bool enabled);

// Internal: Apply direction input for current step
void api_game_apply_input(api_direction_t direction);

// Internal: Execute one game tick
void api_game_tick(void);

#endif // API_GAME_H